import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <model file> <samples file>')
        exit(1)
    checkpoint = sys.argv[1]
    samples_file = sys.argv[2]

    print(f'loading tokenizer from t5-small')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    print(f'loading model from {checkpoint}...')
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=checkpoint)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'model loaded with {params} parameters')

    df = pd.read_csv(samples_file,dtype=str)
    if 'input' not in df.columns:
        print(f'invalid sample file: missing input column in {samples_file} file')
        print(df.head())
        exit()


    def inference(sentence):
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        outputs = model.generate(input_ids,
                                 max_length=10,
                                 num_beams=10,
                                 length_penalty=0.0,
                                 output_scores=True)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)


    print(f'evaluates {df.shape[0]} samples from {samples_file}')

    # df['inference'] = df.apply(lambda t : inference(t['input']), axis=1)
    rows = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['s2_sequence'] == row['s2_sequence']:
            row['inference'] = inference(row['input'])[0]
            if row['inference'] and row['inference'] != np.nan:
                rows.append(row)

    out_df = pd.DataFrame(rows)
    print(f'{out_df.shape[0]} inference rows out of {df.shape[0]} samples')

    print(out_df[['input', 'inference']].head())

    accuracy = accuracy_score(y_true=out_df['s2_sequence'], y_pred=out_df['inference'])
    print(f'accuracy: {accuracy}')
    cm = confusion_matrix(y_true=out_df['s2_sequence'], y_pred=out_df['inference'], normalize='true')
    # cmd = ConfusionMatrixDisplay(cm, display_labels=np.append(out_df['inference'].unique(),'un'))
    cmd = ConfusionMatrixDisplay(cm[:10,:10], display_labels=out_df['inference'].unique()[:10])
    plt.rcParams["figure.figsize"] = (10, 10)
    cmd.plot()
    plt.show()

    output_file = samples_file.replace('.csv', '_infer.csv')
    print(f'save inference results to {output_file}')
    out_df.to_csv(output_file)


