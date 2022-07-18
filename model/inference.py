import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from model.geolabel import cell_id_token2geo

def load_model(checkpoint):
    print(f'loading tokenizer from t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    print(f'loading model from {checkpoint}...')
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=checkpoint)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'model loaded with {params} parameters')
    return tokenizer, model


def inference(tokenizer, model, sentence):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    outputs = model.generate(input_ids,
                             max_length=10,
                             num_beams=10,
                             length_penalty=0.0,
                             output_scores=True,
                             return_dict_in_generate=True
                             )
    cell_id_token = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
    # TODO score = np.argmax(outputs['scores'])
    rect = cell_id_token2geo(cell_id_token)
    ##            0   1    2     3     4     5     6     7     8     9
    ## rect = [x_lo, y_hi, x_hi, y_hi, x_hi, y_lo, x_lo, y_lo, x_lo, y_hi]    [31.81, 31.82, 35.52, 35.53])
    # bbox = y_lo,y_hi, x_lo , x_hi
    bbox = [rect[5], rect[1], rect[0], rect[2]]
    return bbox
