import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
import torch

from datasets import Dataset

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from sklearn import metrics
from transformers import EarlyStoppingCallback



LEARNING_RATE = 2e-05
NUMBER_OF_PARTITIONS = 10
device = torch.device("cpu")
BATCH_SIZE = 4
EPOCHS = 20 * (BATCH_SIZE / 16)
MODEL_NAME = "roberta-base"
TARGET = "stance"
LANGUAGE = "english"
# REP=0
# FOLDS=3
SEQ_LENGTH = 127

col_names = ["tweet", "cn", "offensive", "stance", "informativeness", "felicity"]
data = pd.read_csv("datasets/cn_dataset_{}.csv".format(LANGUAGE), names=col_names)

train_val, test = train_test_split(data, test_size=0.1)

train, dev = train_test_split(train_val, test_size=0.05)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_example(example):
    input_text = example["tweet"] + " [SEP] " + example["cn"]
    tokenized_input = tokenizer(input_text, truncation=True)
    tokenized_input["labels"] = example[TARGET] -1
    return tokenized_input

training_set = Dataset.from_pandas(train).map(tokenize_example)

dev_set = Dataset.from_pandas(dev).map(tokenize_example)

test_set = Dataset.from_pandas(test).map(tokenize_example)

def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    all_true_labels = [str(label) for label in labels]
    all_true_preds = [str(pred) for pred in preds]
    avrge = "macro"
    f1_all = metrics.f1_score(all_true_labels, all_true_preds, average=None, labels=['0','1','2'])

    f1 = metrics.f1_score(all_true_labels, all_true_preds, average=avrge)

    acc = metrics.accuracy_score(all_true_labels, all_true_preds)

    recall = metrics.recall_score(all_true_labels, all_true_preds, average=avrge)

    precision = metrics.precision_score(all_true_labels, all_true_preds, average=avrge)

    confusion_matrix = metrics.confusion_matrix(all_true_labels, all_true_preds)


    w = open("./results_{}_{}_{}-metrics".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE), "a")

    w.write("{},{},{},{}\n".format(str(acc), str(f1), str(precision), str(recall)))
    w.close()

    ans = {
        'accuracy': acc,
        'f1': f1,
        'f1_per_category': f1_all,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': str(confusion_matrix),
    }

    return ans

training_args = TrainingArguments(
        output_dir="./results_cn_eval_{}".format(MODEL_NAME.replace("/", "-")),
        evaluation_strategy="steps",
        eval_steps=10,
        save_total_limit=8,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.05,
        report_to="none",
        metric_for_best_model='f1',
        load_best_model_at_end=True
    )

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= compute_metrics_f1,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    ) 

trainer.train()

results = trainer.predict(test_set)

filename = "./results_test_{}_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME, TARGET, LANGUAGE)

writer = open(filename, "w")
writer.write("{},{},{},{}\n".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"]))
writer.write("{}\n".format(results.metrics["test_f1_per_category"]))
writer.write("{}\n".format(results.metrics["test_confusion_matrix"]))
writer.close()
