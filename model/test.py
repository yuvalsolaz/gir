'''
  Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`
            - `"automatic-speech-recognition"`
            - `"conversational"`
            - `"feature-extraction"`
            - `"fill-mask"`
            - `"image-classification"`
            - `"question-answering"`
            - `"table-question-answering"`
            - `"text2text-generation"`
            - `"text-classification"` (alias `"sentiment-analysis"` available)
            - `"text-generation"`
            - `"token-classification"` (alias `"ner"` available)
            - `"translation"`
            - `"translation_xx_to_yy"`
            - `"summarization"`
            - `"zero-shot-classification"`
            - `"zero-shot-image-classification"`
'''

import sys
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import pipeline

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

    generator = pipeline(task='summarization', model=model, tokenizer=tokenizer)
    df = pd.read_csv(samples_file)
    if 'input' not in df.columns:
        print(f'invalid sample file: missing input column in {samples_file} file')
        print(df.head())
        exit()

    print(f'evaluates {df.shape[0]} samples from {samples_file}')
    df['inference'] = df.apply(lambda t : generator(t['input']), axis=1)
    print(df[['input','inference']].head())

