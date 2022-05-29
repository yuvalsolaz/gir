'''
    downstream task:
    classify region label
    1. load dataset with cell labels
    2. tokenize text field
    3. labels => int
    4. train

'''

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer
import datasets
import numpy as np


label_field = 'labels'

def unique_labels(ds):
    ds.set_format('pandas')
    unique_labels = ds[label_field].unique()
    ds.reset_format()
    return unique_labels

def model_summary(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def train(dataset_path, checkpoint):
    print (f'load dataset from: {dataset_path}')
    ds = datasets.load_from_disk(dataset_path=dataset_path)
    train = ds['train']
    test = ds['test']
    print(f'{train.shape[0]} train samples')
    print(f'{test.shape[0]} test samples')

    print(f'loading tokenizer & model from {checkpoint} checkpoint')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    all_labels = np.array(list(set.union(set(unique_labels(train)), set(unique_labels(test)))))
    # label2id = {k: np.where(all_labels == k)[0][0] for k in all_labels}
    # id2label = {np.where(all_labels == k)[0][0]: k for k in all_labels}

    print(f'loading model from {checkpoint} with {len(all_labels)} labels')
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               # id2label=id2label,
                                                               # label2id=label2id,
                                                               num_labels=len(all_labels))

    params = model_summary(model)
    print(f'model loaded with {params} parameters')

    def concat_fields(samples):
        english_label = samples["english_label"] if samples["english_label"] is not None else ''
        english_desc = samples["english_desc"] if samples["english_desc"] is not None else ''
        return {'text': f'{english_label} {english_desc}'}

    # train = train.map(concat_fields, batched=False)
    # test = test.map(concat_fields, batched=False)

    train = train.map(lambda x: {'text': x["english_desc"] if x["english_desc"] is not None else ''})
    test  = test.map( lambda x: {'text': x["english_desc"] if x["english_desc"] is not None else ''})


    def tokenize_function(samples):
        return tokenizer(samples["text"] , padding=True, truncation=True)

    print (f'tokenize train...')
    tokenized_train = train.map(tokenize_function, batched=True)

    print(f'tokenize test...')
    tokenized_test = test.map(tokenize_function, batched=True)

    non_label_columns = train.column_names
    non_label_columns.remove('labels')
    print(f'remove all columns except the label: {non_label_columns}')
    tokenized_train = tokenized_train.remove_columns(non_label_columns)
    tokenized_test = tokenized_test.remove_columns(non_label_columns)


    training_args = TrainingArguments(output_dir='trainer',
                                      report_to=['tensorboard'],
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=32,
                                      num_train_epochs=20,
                                      logging_steps=50,
                                      save_steps=500,
                                      save_total_limit=3,
                                      evaluation_strategy='steps',
                                      eval_steps=5000,
                                      no_cuda=False)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_train = tokenized_train.shuffle(seed=7)
    tokenized_test = tokenized_test.shuffle(seed=5)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator
    )

    print(f'training...{training_args}')

    trainer.train()


import sys
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset path>')
        exit(1)
    dataset_path = sys.argv[1]
    checkpoint = 'roberta-base'
    if len(sys.argv) > 2:
        checkpoint = sys.argv[2]

    train(dataset_path=dataset_path, checkpoint=checkpoint)
