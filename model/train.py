'''
    downstream task:
    classify region label
    1. load dataset with cell labels
    2. tokenize text field
    3. labels => int
    4. train

'''

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import datasets
import numpy as np
from nltk.tokenize import sent_tokenize

from datasets import load_metric
import s2geometry as s2
rouge_score = load_metric("rouge")

from geolabel import label_field

max_input_length = 512
max_target_length = 30


def unique_labels(ds):
    ds.set_format('pandas')
    unique_labels = ds[label_field].unique()
    print(f'{label_field} value counts:\n{ds[label_field].value_counts()}')
    ds.reset_format()
    return unique_labels


def model_summary(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def train(dataset_path, checkpoint):
    print(f'load dataset from: {dataset_path}')
    ds = datasets.load_from_disk(dataset_path=dataset_path)
    train = ds['train']
    test = ds['test']
    all_labels = np.array(list(set.union(set(unique_labels(train)), set(unique_labels(test)))))
    print(f'{train.shape[0]} train samples {test.shape[0]} test samples with {len(all_labels)} unique labels')

    print(f'loading tokenizer from t5-base checkpoint')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    print(f'loading model from {checkpoint}')
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=checkpoint)
    params = model_summary(model)
    print(f'model loaded with {params} parameters')

    print(f'tokenize input and labels on train')
    # train = train.map(lambda x: {'text': x["english_desc"] if x["english_desc"] is not None else ''})
    # test = test.map(lambda x: {'text': x["english_desc"] if x["english_desc"] is not None else ''})
    def preprocess_function(sample):
        model_inputs = tokenizer(
            sample['english_desc'], max_length=max_input_length, truncation=True
        )
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                sample[label_field], max_length=max_target_length, truncation=True
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    non_label_columns = train.column_names
    tokenized_train = train.map(preprocess_function, batched=True)
    print(f'tokenize input and labels on test')
    tokenized_test = test.map(preprocess_function, batched=True)

    print(f'remove all columns except input_ids labels and mask:\n {non_label_columns}')
    tokenized_train = tokenized_train.remove_columns(non_label_columns)
    tokenized_test = tokenized_test.remove_columns(non_label_columns)

    training_args = Seq2SeqTrainingArguments(output_dir='seq2seq',
                                             report_to=['tensorboard'],
                                             learning_rate=1.5e-5,
                                             per_device_train_batch_size=12,
                                             per_device_eval_batch_size=12,
                                             weight_decay=0.01,
                                             num_train_epochs=20,
                                             predict_with_generate=True,
                                             logging_steps=50,
                                             load_best_model_at_end=True,
                                             save_steps=50000,
                                             save_total_limit=3,
                                             evaluation_strategy='steps',
                                             eval_steps=50000,
                                             no_cuda=False)

    tokenized_train = tokenized_train.shuffle(seed=7)
    tokenized_test = tokenized_test.shuffle(seed=5)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # features = [tokenized_train[i] for i in range(2)]
    # data_collator(features)

    def compute_rouge_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    # def iou_score(predictions, references):
    #     score = 1.0
    #     oops = False
    #     for idx, c in erumerate(label):
    #         if oops:
    #             score /= 4.0  # TODO : log
    #         if c != pred[idx]
    #             oops = True
    #     return score

    def compute_geo_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # TODO: Compute avg IOU between predicted cell and labeled cell as scores
        result = None # iou_score(predictions=decoded_preds, references=decoded_labels)
        # Extract the median scores
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}


    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_rouge_metrics
    )

    print(f'training...{training_args}')

    trainer.train()


import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset path>')
        exit(1)
    dataset_path = sys.argv[1]
    checkpoint = 't5-base'
    if len(sys.argv) > 2:
        checkpoint = sys.argv[2]

    train(dataset_path=dataset_path, checkpoint=checkpoint)
