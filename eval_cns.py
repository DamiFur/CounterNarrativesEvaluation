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
LANGUAGE = args.language
SEQ_LENGTH = 127
target = args.category
model_name_adapted = MODEL_NAME.replace("/", "-")
extension = "_extended" if args.extended else ""


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

def tokenize_example(example):
    input_text = example["tweet"] + " [SEP] " + example["cn"]
    return tokenizer(input_text, truncation=True)

# If the folder does not exist, create it
if not os.path.exists("predictions"):
    os.makedirs("predictions")

results = {}

four_labels = target == "informativeness" and args.test_zero
test_zero = "_test_zero" if four_labels else ""
num_labels = 4 if four_labels else 3

id2label = {0: "0", 1: "1", 2: "2"} if not four_labels else {0: "0", 1: "1", 2: "2", 3: "3"}
label2id = {"0": 0, "1": 1, "2": 2} if not four_labels else {"0": 0, "1": 1, "2": 2, "3": 3}
pretrained_model_name = "{}_{}_{}_{}{}{}".format(LEARNING_RATE, model_name_adapted, target, LANGUAGE, extension, test_zero)
model = AutoModelForSequenceClassification.from_pretrained(f"models/{pretrained_model_name}", num_labels=num_labels, id2label = id2label, label2id = label2id)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


# traverse all folders on 'counter-narratives' folder
for folder in os.listdir("counter-narratives"):
    splitted_folder = folder.split("_")
    model = splitted_folder[1]
    language = splitted_folder[2]
    learning_strategy = splitted_folder[3]
    arg_info = splitted_folder[4]
    cn_strategy = splitted_folder[5]
    key = f"{model}-{language}-{learning_strategy}-{arg_info}-{cn_strategy}"
    if key not in results:
        results[key] = []

    w = open("predictions/" + folder + "_predictions", "w")

    # traverse all files in the folder
    for file in os.listdir("counter-narratives/" + folder):
        # read the first line from file
        with open("counter-narratives/" + folder + "/" + file, "r") as f:
            # read the first line from file
            print(folder)
            print(file)
            hate_tweet = f.readline()
            # print(line)
            cn = f.readline()
            print(cn)
            prediction = classifier(hate_tweet + " [SEP] " + cn)
            w.write(hate_tweet + "\t" + cn + "\t" + str(prediction) + "\n")
            results[key].append(prediction)
    w.close()

output = open(f"predictions/results_{target}.tsv", 'w')

for key in results:
    output.write(key.replace("-", "\t") + "\t" + str(results[key]) + "\n")
output.close()