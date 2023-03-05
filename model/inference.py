import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


def load_model(checkpoint):
    print(f'loading tokenizer from t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base', model_max_length=512)

    print(f'loading model from {checkpoint}...')
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=checkpoint)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'model loaded with {params} parameters')
    return tokenizer, model


def inference(tokenizer, model, sentence):
    input_ids = tokenizer(sentence.lower(), return_tensors="pt").input_ids
    outputs = model.generate(input_ids,
                             max_length=10,
                             num_beams=10,
                             # num_return_sequences=10,
                             length_penalty=0.0,
                             output_scores=True,
                             return_dict_in_generate=True)
    cellid = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices,
                                                        normalize_logits=True)
    seq_len = transition_scores[0].shape[-1] + 1
    token_scores = [np.prod(np.exp(transition_scores[0][:seq_len]).numpy()) for seq_len in np.arange(1, seq_len)]
    return cellid, token_scores
