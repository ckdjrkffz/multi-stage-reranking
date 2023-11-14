import pandas as pd
import os
import random
from tqdm import tqdm
import argparse
import json

from transformers import (
    AutoTokenizer,
)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="dataset")
parser.add_argument('--output_data_path', type=str, default="dataset/processed_data")
parser.add_argument("--model_name_or_path", type=str, default=None, help="model name or path")

args = parser.parse_args()

random.seed(0)

#####################path################################

document_path = f"{args.data_path}/corpus.jsonl"
query_path = f"{args.data_path}/queries.jsonl"

train_qrel_path = f"{args.data_path}/qrels/train.tsv"
dev_qrel_path = f"{args.data_path}/qrels/dev.tsv"
test_qrel_path = f"{args.data_path}/qrels/test.tsv"


document_output_path = f"{args.output_data_path}/document.json"
query_output_path = f"{args.output_data_path}/query.json"

train_qrel_output_path = f"{args.output_data_path}/qrels/train.json"
dev_qrel_output_path = f"{args.output_data_path}/qrels/dev.json"
test_qrel_output_path = f"{args.output_data_path}/qrels/test.json"


os.makedirs(args.output_data_path, exist_ok=True)
os.makedirs(f"{args.output_data_path}/qrels", exist_ok=True)

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

######################document#######################

#{"_id":..., "text":...}
document = []
with open(document_path)as f:
    for l in f:
        document.append(json.loads(l))

document = \
    {str(dic["_id"]):
        {
            "text":dic["text"],
            "text_tokenized":tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dic["text"]))
        }
    for dic in tqdm(document)}

with open(document_output_path, "w")as f:
    json.dump(document, f)


######################query##############################

#{"_id":..., "text":...}
query= []
with open(query_path)as f:
    for l in f:
        query.append(json.loads(l))

query = \
    {str(dic["_id"]):
        {
            "text":dic["text"],
            "text_tokenized":tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dic["text"]))
        }
    for dic in tqdm(query)}

with open(query_output_path, "w")as f:
    json.dump(query, f)

###################qrel###########################

def qrel_process(qrel_path, qrel_output_path):
    if os.path.exists(qrel_path)==False:
        print(f"Thre is no {qrel_path}")
    else:
        qrel = pd.read_csv(qrel_path, sep="\t")
        qrel_processed = {}
        for query_id, doc_id in tqdm(zip(qrel["query-id"].tolist(), qrel["corpus-id"].tolist())):
            query_id, doc_id = str(query_id), str(doc_id)
            if query_id not in qrel_processed:
                qrel_processed[query_id] = {"positive_doc_id":[]}
            qrel_processed[query_id]["positive_doc_id"].append(doc_id)
        with open(qrel_output_path, "w")as f:
            json.dump(qrel_processed, f)


qrel_process(train_qrel_path, train_qrel_output_path)
qrel_process(dev_qrel_path, dev_qrel_output_path)
qrel_process(test_qrel_path, test_qrel_output_path)
