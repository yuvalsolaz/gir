'''
    downstream task:
    classify region label
    1. load dataset with cell labels
    2. tokenize text field
    3. labels => int
    4. train

'''
import random

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import datasets
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, Trainer

# TODO: split in geolabel save to disk
raw_dataset_train = datasets.load_from_disk('/home/yuvalso/repository/gir/data/all_items')
raw_dataset_eval = datasets.load_from_disk('/home/yuvalso/repository/gir/data/all_items')
raw_dataset_test = datasets.load_from_disk('/home/yuvalso/repository/gir/data/all_items')

checkpoint = 'roberta-base'
num_labels  = 6 # TODO: calculate # of labels
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
# TODO : note - labels must be special int (see example vit)

training_args = TrainingArguments(output_dir='geo_dict')

def tokenize_function(samples):
    # samples['label'] = TODO:
    return tokenizer(samples["english_desc"], truncation=True)


tokenized_datasets_train = raw_dataset_train.map(tokenize_function, batched=True)
tokenized_datasets_eval = raw_dataset_eval.map(tokenize_function, batched=True)
tokenized_datasets_test = raw_dataset_test.map(tokenize_function, batched=True)


tokenized_datasets_train = tokenized_datasets_train.remove_columns(["cell_id", "labels"])
tokenized_datasets_eval = tokenized_datasets_eval.rename_column("cell_id", "labels")
tokenized_datasets_test = tokenized_datasets_test.rename_column("cell_id", "labels")

tokenized_datasets_train = tokenized_datasets_train.rename_column("cell_id", "labels")
tokenized_datasets_eval = tokenized_datasets_eval.rename_column("cell_id", "labels")
tokenized_datasets_test = tokenized_datasets_test.rename_column("cell_id", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


train_dataloader = DataLoader(
    tokenized_datasets_train, shuffle=True, batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()