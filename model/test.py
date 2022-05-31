import numpy as np
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline
from transformers import AutoTokenizer

def inference():

import sys
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <model file> <samples file>')
        exit(1)
    checkpoint = sys.argv[1]
    samples_file = sys.argv[2]
    tokenizer_pre
    print(f'loading tokenizer from 'roberta-base'')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    print(f'loading model from {checkpoint}...')
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint) #  num_labels=len(all_labels))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'model loaded with {params} parameters')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')


