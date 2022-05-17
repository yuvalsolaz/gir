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

label_field = 'cell_id'

def labels_count(dataset):
    dataset.set_format('pandas')
    labels_count = len(dataset[label_field].unique())
    dataset.reset_format()
    return labels_count

def train(dataset_path):
    print (f'load dataset from: {dataset_path}')
    ds = datasets.load_from_disk(dataset_path=dataset_path)
    train = ds['train']
    test = ds['test']
    print(f'{train.shape[0]} train samples')
    print(f'{test.shape[0]} test samples')

    test_num_labels  = labels_count(test)
    print(f'test number of labels: {test_num_labels}')
    train_num_labels  = labels_count(train)
    print(f'train number of labels: {train_num_labels}')

    checkpoint = 'roberta-base'
    print(f'loading tokenizer & model from {checkpoint} checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=train_num_labels)

    def tokenize_function(samples):
        return tokenizer(samples["english_desc"], truncation=True)

    print (f'tokenize train...')
    tokenized_train = train.map(tokenize_function, batched=True)

    print(f'tokenize test...')
    tokenized_test = test.map(tokenize_function, batched=True)

    non_label_columns = train.column_names
    non_label_columns.remove(label_field)
    print(f'remove all columns except the label: {non_label_columns}')
    tokenized_train = tokenized_train.remove_columns(non_label_columns)
    tokenized_test = tokenized_test.remove_columns(non_label_columns)

    print('mapping labels from float to uint64 : TODO: move to geo label module ')
    max_np_int = np.iinfo(np.uint64).max  # 2**64 - 1 = 18446744073709551615
    def to_uint(sample):
        try:
            cid = np.uint64(sample['cell_id'])
            return {'labels': cid}
        except Exception as ex:
            return {'labels': None}


    tokenized_train = tokenized_train.map(to_uint)
    tokenized_test = tokenized_test.map(to_uint)
    # test = tokenized_train.filter(lambda x: x['labels'] is None)
    # print(f'{test.shape[0]} None samples out of {tokenized_train.shape[0]}')
    #tokenized_test = tokenized_test.rename_column('cell_id', 'labels')


    print('training...')
    training_args = TrainingArguments(output_dir='trainer', evaluation_strategy='epoch')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator
    )
    trainer.train()


import sys
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset path>')
        exit(1)
    dataset_path = sys.argv[1]
    train(dataset_path=dataset_path)
