import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os

parser = argparse.ArgumentParser(description="Train models for automatic evaluation of counter-narratives")
parser.add_argument('--lr', type=float, default=2e-05, help="Learning rate for training the model. Default value is 2e-05")
parser.add_argument("--model_name", type=str, default="roberta-base")
parser.add_argument("--category", type=str, default="offensive")
parser.add_argument("--language", type=str, default="english")
parser.add_argument("--extended", type=bool, default=False)
parser.add_argument("--test_zero", type=bool, default=False, help="Only used when predicting Informativeness, if True, we consider predictions on examples that had 0 as their Informativeness value")

args = parser.parse_args()

LEARNING_RATE = args.lr
BATCH_SIZE = 4
EPOCHS = 20 * (BATCH_SIZE / 16)
MODEL_NAME = args.model_name
TARGET = args.category
LANGUAGE = args.language
SEQ_LENGTH = 127
four_labels = args.test_zero and args.category == "informativeness"
extension = "_extended" if args.extended else ""
test_zero = "_test_zero" if four_labels else ""
extension = "_extended" if args.extended else ""
test_zero = "_test_zero" if four_labels else ""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
num_labels = 4 if four_labels else 3
model_name_adapted = MODEL_NAME.replace("/", "-")
pretrained_model_name = "{}-{}-{}-{}{}{}".format(model_name_adapted, TARGET, LANGUAGE, LEARNING_RATE, extension, test_zero)
model = AutoModelForSequenceClassification.from_pretrained(f"models/{pretrained_model_name}", num_labels=num_labels)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def tokenize_example(example):
    input_text = example["tweet"] + " [SEP] " + example["cn"]
    return tokenizer(input_text, truncation=True)

# traverse all folders on 'counter-narratives' folder
for folder in os.listdir("counter-narratives"):
    # traverse all files in the folder
    for file in os.listdir("counter-narratives/" + folder):
        # read the first line from file
        with open("counter-narratives/" + folder + "/" + file) as f:
            # read the first line from file
            print(folder)
            print(file)
            hate_tweet = f.readline()
            # print(line)
            cn = f.readline()
            print(cn)
            print(classifier(hate_tweet + " [SEP] " + cn))