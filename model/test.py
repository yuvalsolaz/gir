import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <model file> <dataset_path>')
        exit(1)
    checkpoint = sys.argv[1]
    dataset_path = sys.argv[2]

    print(f'load dataset from: {dataset_path}')
    ds = datasets.load_from_disk(dataset_path=dataset_path)
    test = ds['train'].select(range(1000))

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
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f'evaluates {test.shape[0]} samples from {dataset_path}')

    test = test.map(lambda x: {'inference': inference(x["english_desc"]) if x["english_desc"] is not None else ''})
    accuracy = accuracy_score(y_true=test['s2_sequence'], y_pred=test['inference'])
    print(f'\n accuracy: {accuracy}')
    # save to disk
    output_path = 'inference.csv'
    print(f'save dataset to: {output_path}')
    test.set_format('pandas')
    test.to_csv(output_path)



