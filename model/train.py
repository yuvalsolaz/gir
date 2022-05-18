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


def train(dataset_path):
    print (f'load dataset from: {dataset_path}')
    ds = datasets.load_from_disk(dataset_path=dataset_path)
    train = ds['train']
    test = ds['test']
    print(f'{train.shape[0]} train samples')
    print(f'{test.shape[0]} test samples')

    checkpoint = 'roberta-base'
    print(f'loading tokenizer & model from {checkpoint} checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    all_labels = np.array(list(set.union(set(unique_labels(train)), set(unique_labels(test)))))
    print(f'loading model from {checkpoint} with {len(all_labels)} labels')
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               # id2label=id2label,
                                                               # label2id=label2id,
                                                               num_labels=len(all_labels))

    def tokenize_function(samples):
        return tokenizer(samples["english_desc"], padding=True, truncation=True)

    print (f'tokenize train...')
    tokenized_train = train.map(tokenize_function, batched=True)

    print(f'tokenize test...')
    tokenized_test = test.map(tokenize_function, batched=True)

    non_label_columns = train.column_names
    non_label_columns.remove('labels')
    print(f'remove all columns except the label: {non_label_columns}')
    tokenized_train = tokenized_train.remove_columns(non_label_columns)
    tokenized_test = tokenized_test.remove_columns(non_label_columns)


    print('training...')
    training_args = TrainingArguments(output_dir='trainer', evaluation_strategy='epoch', no_cuda=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# try batch
    #samples = tokenized_train[:8]
    # samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
    #[len(x) for x in samples["input_ids"]]
    #samples = samples.remove_columns('cell_id')

    # train_dataloader = DataLoader(
    #     tokenized_train, shuffle=True, batch_size=8, collate_fn=data_collator
    # )
    #batch = data_collator(samples)
    #{k: v.shape for k, v in batch.items()}
    #model(batch)
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        tokenized_train, shuffle=True, batch_size=8, collate_fn=data_collator
    )
    for batch in train_dataloader:
        break
    {k: v.shape for k, v in batch.items()}
# region batch

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
