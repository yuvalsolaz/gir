import sys
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import datasets

'''
Pi predicted class for each test example i, and all of its ancestor classes (car, pickup, isuzu)    
Ti true class of test example i, and all its ancestor classes

Definitions for hierarchical precision hP
    hP(predict,) = sum(Pi intersect Ti) / sum(Pi)  

Definitions for hierarchical recall hR
    hR(predict,) = sum(Pi intersect Ti) / sum(Ti)  

Definitions for hierarchical f-measure hF
    hF = 2 * hP * hR / (hP + hR)
'''

def hirarchies_metric(y_true, y_pred):

    def intersect(true_seq, pred_seq):
        for i in range(min(len(true_seq),len(pred_seq))):
            if true_seq[i] != pred_seq[i]:
                break
        return true_seq[:i+1]

    x_array = np.array(list(zip(y_true,y_pred)))

    hA_seq = [intersect(sample[0], sample[1]) for sample in x_array]

    vlen = np.vectorize(len)
    Ti = vlen(x_array[0])
    hA = vlen(hA_seq)
    haccuracy = np.divide(hA, sum(Ti))
    return haccuracy




if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <model file> <dataset_path>')
        exit(1)
    checkpoint = sys.argv[1]
    dataset_path = sys.argv[2]

    print(f'load dataset from: {dataset_path}')
    ds = datasets.load_from_disk(dataset_path=dataset_path)
    test = ds['train'].select(range(10))

    print(f'loading tokenizer from t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    print(f'loading model from {checkpoint}...')
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=checkpoint)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'model loaded with {params} parameters')

    def inference(sentence):
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        outputs = model.generate(input_ids,
                                 max_length=10,
                                 num_beams=10,
                                 length_penalty=0.0,
                                 output_scores=True,
                                 return_dict_in_generate=True
                                 )
        end_of_sequence = '\x00'
        return tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0] + end_of_sequence

    print(f'evaluates {test.shape[0]} samples from {dataset_path}')

    test = test.map(lambda x: {'inference': inference(x["english_desc"]) if x["english_desc"] is not None else ''})
    accuracy = accuracy_score(y_true=test['s2_sequence'], y_pred=test['inference'])
    h_accuracy = hirarchies_metric(y_true=test['s2_sequence'], y_pred=test['inference'])
    print(f'\n accuracy: {accuracy} \n hierarchy accuracy: {h_accuracy}')
    # save to disk
    output_path = 'inference.csv'
    print(f'save dataset to: {output_path}')
    test.set_format('pandas')
    test.to_csv(output_path)



