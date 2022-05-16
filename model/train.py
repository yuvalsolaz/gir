'''
    downstream task:
    classify region label
    1. load dataset with cell labels
    2. tokenize text field
    3. labels => int
    4. train

'''

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer
from torch.utils.data import DataLoader
import datasets

label_field = 'cell_id'

def labels_count(dataset):
    dataset.set_format('pandas')
    labels_count = len(dataset[label_field].unique())
    dataset.reset_format()
    return labels_count

def train(dataset_path):
    dataset = datasets.load_from_disk(dataset_path=dataset_path)
    train = dataset['train']
    test = dataset['test']
    non_label_columns = train.column_names
    non_label_columns.remove(label_field)
    checkpoint = 'roberta-base'
    num_labels  = labels_count(train)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    training_args = TrainingArguments(output_dir='geo_dict')

    def tokenize_function(samples):
        return tokenizer(samples["english_desc"], truncation=True)

    tokenized_train = train.map(tokenize_function, batched=True)
    tokenized_test = test.map(tokenize_function, batched=True)

    # remove all columns except the label:
    tokenized_train = tokenized_train.remove_columns(non_label_columns)
    tokenized_test = tokenized_test.remove_columns(non_label_columns)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    train_dataloader = DataLoader(
        tokenized_train, shuffle=True, batch_size=8, collate_fn=data_collator
    )

    for batch in train_dataloader:
        break
    {k: v.shape for k, v in batch.items()}
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)
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
