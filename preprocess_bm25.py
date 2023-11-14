import os
import random
from tqdm import tqdm
import argparse
import json
import subprocess

from pyserini.search.lucene import LuceneSearcher

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="dataset")
parser.add_argument('--output_data_path', type=str, default="dataset/processed_data")
parser.add_argument("--model_name_or_path", type=str, default=None, help="model name or path")
parser.add_argument("--bm25_num_candidate", type=int, default=100)

args = parser.parse_args()

random.seed(0)

#####################path################################

document_path = f"{args.data_path}/document.json"
query_path = f"{args.data_path}/query.json"

train_qrel_path = f"{args.data_path}/qrels/train.json"
dev_qrel_path = f"{args.data_path}/qrels/dev.json"
test_qrel_path = f"{args.data_path}/qrels/test.json"

document_processed_path = f"{args.output_data_path}/document_processed"
index_path = f"{args.output_data_path}/index"

train_qrel_output_path = f"{args.output_data_path}/qrels/train.json"
dev_qrel_output_path = f"{args.output_data_path}/qrels/dev.json"
test_qrel_output_path = f"{args.output_data_path}/qrels/test.json"


os.makedirs(args.output_data_path, exist_ok=True)
os.makedirs(f"{args.output_data_path}/document_processed", exist_ok=True)
os.makedirs(f"{args.output_data_path}/index", exist_ok=True)
os.makedirs(f"{args.output_data_path}/qrels", exist_ok=True)


####################index##################################

with open(document_path)as f:
    document = json.load(f)

with open(query_path)as f:
    query = json.load(f)

document_processed = [
    {"id": doc_id, "contents": value["text"]} for doc_id, value in document.items()
]

with open(f"{document_processed_path}/document_processed.json", "w")as f:
    json.dump(document_processed, f)

subprocess.run(
    f"""
    python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input {document_processed_path} \
    --index {index_path} \
    --generator DefaultLuceneDocumentGenerator \
    --threads 1 \
    """.split()
)

bm25_searcher = LuceneSearcher(index_path)
bm25_searcher.set_bm25(0.9, 0.4)


###################qrel###########################

def qrel_process(qrel_path, qrel_output_path):
    if os.path.exists(qrel_path)==False:
        print(f"There is no {qrel_path}")
    else:
        with open(qrel_path)as f:
            qrel = json.load(f)

        for query_id in tqdm(qrel.keys()):

            query_text = query[query_id]["text"]
            bm25_score_list = bm25_searcher.search(query_text, k=args.bm25_num_candidate)
            bm25_score_list = \
                sorted([(value.score, value.docid) \
                    for value in bm25_score_list], key=lambda x: -x[0])
            bm25_doc_id = \
                [doc_id for _, doc_id in bm25_score_list]
            qrel[query_id]["bm25_doc_id"] = bm25_doc_id

        with open(qrel_output_path, "w")as f:
            json.dump(qrel, f)


qrel_process(train_qrel_path, train_qrel_output_path)
qrel_process(dev_qrel_path, dev_qrel_output_path)
qrel_process(test_qrel_path, test_qrel_output_path)