
import random
import time
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
import json
import io,sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextDataset,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup,
)

import pytrec_eval


class ScorePredictor():

    def __init__(
        self,
        args,
        id2doc,
        id2query,
    ):

        self.id2doc = id2doc
        self.id2query = id2query

        if args.use_bert:
            # config
            config = AutoConfig.from_pretrained(args.model_name_or_path[0], cache_dir=None)
            config.num_labels = 2

            # tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path[0])

            # model
            print(f"model path: {args.model_name_or_path}")
            self.model = []
            for path in args.model_name_or_path:
                model = AutoModelForSequenceClassification.from_pretrained(
                    path,
                    config=config,
                )
                model.resize_token_embeddings(len(self.tokenizer))
                model=model.to(args.device)
                model.eval()
                self.model.append(model)

            print(f"model size: {len(self.model)}")

            self.bert_task_type = args.bert_task_type
            self.source_block_size = args.source_block_size
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
            self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            self.device = args.device
            self.batch_size = args.batch_size

        # second model
        if args.use_second_bert:
            # config
            config = AutoConfig.from_pretrained(args.second_model_name_or_path[0], cache_dir=None)
            config.num_labels = 2

            print(f"second model path: {args.second_model_name_or_path}")
            self.second_model = []
            for path in args.second_model_name_or_path:
                model = AutoModelForSequenceClassification.from_pretrained(
                    path,
                    config=config,
                )
                model = model.to(args.device)
                model.eval()
                self.second_model.append(model)

            self.second_bert_task_type = args.second_bert_task_type
            self.second_source_block_size = args.second_source_block_size

            print(f"second model size: {len(self.second_model)}")

    def bert_scorer(self, query_id, cand_doc_id, doc2index, model_num):

        query_text_tokenized = self.id2query[query_id]["text_tokenized"]

        text_list = []
        attention_mask_list = []
        index_list = []

        predict_prob = np.zeros(len(doc2index), dtype="float")

        models = \
            self.model if model_num == "first" else \
            self.second_model
        bert_task_type = \
            self.bert_task_type if model_num == "first" else \
            self.second_bert_task_type if model_num == "second" else None
        source_block_size = \
            self.source_block_size if model_num == "first" else \
            self.second_source_block_size if model_num == "second" else None

        if bert_task_type == "classification":
            for doc_id in cand_doc_id:
                doc_text_tokenized = self.id2doc[doc_id]["text_tokenized"]
                index = doc2index[doc_id]

                text = \
                    query_text_tokenized + \
                    [self.sep_token_id] + \
                    doc_text_tokenized
                text = text[:source_block_size-2]
                pad_size = source_block_size - 2 - len(text)

                text = \
                    [self.bos_token_id] + \
                    text + \
                    [self.eos_token_id] + \
                    [self.pad_token_id] * pad_size

                attention_mask = \
                    [1] * (source_block_size - pad_size) + \
                    [0] * pad_size

                text_list.append(text)
                attention_mask_list.append(attention_mask)
                index_list.append(index)

            for start in range(0, len(text_list), self.batch_size):
                with torch.no_grad():
                    text = text_list[start:start+self.batch_size]
                    text = torch.tensor(text, dtype=torch.long).to(self.device)

                    attention_mask = attention_mask_list[start:start+self.batch_size]
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

                    index = index_list[start:start+self.batch_size]

                    outputs = []  # (num_models, batch_size)
                    for model in models:
                        output = model(input_ids=text, attention_mask=attention_mask)[0]
                        output = output.softmax(dim=-1)[:,1].cpu().numpy()
                        outputs.append(output)
                    predict_prob[index] = np.average(outputs, axis=0)  # (batch_size)

        elif bert_task_type == "pairwise":
            for doc_id1 in cand_doc_id:
                for doc_id2 in cand_doc_id:
                    if doc_id1 == doc_id2:
                        continue
                    doc_text_tokenized1 = self.id2doc[doc_id1]["text_tokenized"]
                    doc_text_tokenized2 = self.id2doc[doc_id2]["text_tokenized"]
                    index1 = doc2index[doc_id1]
                    index2 = doc2index[doc_id2]

                    length_size = \
                        len(query_text_tokenized) + \
                        len(doc_text_tokenized1) + \
                        len(doc_text_tokenized2) + \
                        1 + 1
                    over_size = length_size + 2 - source_block_size
                    if over_size > 0:
                        doc_text_tokenized1 = doc_text_tokenized1[:-int((over_size+1)/2)]
                        doc_text_tokenized2 = doc_text_tokenized2[:-int((over_size+1)/2)]

                    text = \
                        query_text_tokenized + \
                        [self.sep_token_id] + \
                        doc_text_tokenized1 + \
                        [self.sep_token_id] + \
                        doc_text_tokenized2
                    text = text[:source_block_size-2]
                    pad_size = source_block_size - 2 - len(text)

                    text = \
                        [self.bos_token_id] + \
                        text + \
                        [self.eos_token_id] + \
                        [self.pad_token_id] * pad_size

                    attention_mask = \
                        [1] * (source_block_size - pad_size) + \
                        [0] * pad_size

                    text_list.append(text)
                    attention_mask_list.append(attention_mask)
                    index_list.append((index1, index2))

            index2prob = defaultdict(list)
            for start in range(0, len(text_list), self.batch_size):
                with torch.no_grad():
                    text = text_list[start:start+self.batch_size]
                    text = torch.tensor(text, dtype=torch.long).to(self.device)

                    attention_mask = attention_mask_list[start:start+self.batch_size]
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

                    index = index_list[start:start+self.batch_size]

                    outputs = []  # (num_models, batch_size)
                    for model in models:
                        output = model(input_ids=text, attention_mask=attention_mask)[0]
                        output = output.softmax(dim=-1)[:,0].cpu().numpy()
                        outputs.append(output)
                    outputs = np.average(outputs, axis=0) #(batch_size)
                    for (ii1, ii2), oo in zip(index, outputs):
                        index2prob[ii1].append(oo)
                        index2prob[ii2].append(1-oo)
            for index, outputs in index2prob.items():
                if len(outputs) != 2 * (len(cand_doc_id)-1):
                    print(index, len(outputs), len(cand_doc_id))
                    exit()
                predict_prob[index] = np.average(outputs)

        elif bert_task_type == "random":
            for doc_id1 in cand_doc_id:
                for doc_id2 in cand_doc_id:
                    if doc_id1 == doc_id2:
                        continue
                    doc_text_tokenized1 = self.id2doc[doc_id1]["text_tokenized"]
                    doc_text_tokenized2 = self.id2doc[doc_id2]["text_tokenized"]
                    index = doc2index[doc_id1]

                    text = \
                        query_text_tokenized + \
                        [self.sep_token_id] + \
                        doc_text_tokenized1 + \
                        [self.sep_token_id] + \
                        doc_text_tokenized2
                    text = text[:source_block_size-2]
                    pad_size = source_block_size - 2 - len(text)

                    text = \
                        [self.bos_token_id] + \
                        text + \
                        [self.eos_token_id] + \
                        [self.pad_token_id] * pad_size

                    attention_mask = \
                        [1] * (source_block_size - pad_size) + \
                        [0] * pad_size

                    text_list.append(text)
                    attention_mask_list.append(attention_mask)
                    index_list.append(index)

            index2prob = defaultdict(list)
            for start in range(0, len(text_list), self.batch_size):
                with torch.no_grad():
                    text = text_list[start:start+self.batch_size]
                    text = torch.tensor(text, dtype=torch.long).to(self.device)

                    attention_mask = attention_mask_list[start:start+self.batch_size]
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

                    index = index_list[start:start+self.batch_size]

                    outputs = [] #(num_models, batch_size)
                    for model in models:
                        output = model(input_ids=text, attention_mask = attention_mask)[0]
                        output = output.softmax(dim=-1)[:,0].cpu().numpy()
                        outputs.append(output)
                    outputs = np.average(outputs, axis=0) #(batch_size)
                    for ii, oo in zip(index, outputs):
                        index2prob[ii].append(oo)
            for index, outputs in index2prob.items():
                if len(outputs)!=len(cand_doc_id)-1:
                    print(index, len(outputs), len(cand_doc_id))
                    exit()
                predict_prob[index] = random.random()*10
        else:
            print("No exist task")
            exit()

        return predict_prob

    def predict(self, query_id, cand_doc_id):

        doc2index = {doc_id:index for index, doc_id in enumerate(cand_doc_id)}
        predict_prob = np.zeros(len(cand_doc_id))
        for score, doc_id in enumerate(cand_doc_id[::-1]):
            predict_prob[doc2index[doc_id]] =  score

        #BM25
        predict_doc_id_sorted = \
            sorted([(score, cand_doc_id[index]) \
                for index,score in enumerate(predict_prob)], key=lambda x: -x[0])
        predict_doc_id_highest = \
            [doc_id for _, doc_id in predict_doc_id_sorted[:args.bert_num_candidate]]

        #BERT
        if args.use_bert:
            predict_prob = self.bert_scorer(query_id, predict_doc_id_highest, doc2index, "first")

            predict_doc_id_sorted = \
                sorted([(score, cand_doc_id[index]) \
                    for index,score in enumerate(predict_prob)], key=lambda x: -x[0])
            predict_doc_id_highest = \
                [doc_id for _, doc_id in predict_doc_id_sorted[:args.second_bert_num_candidate]]

        #Second BERT
        if args.use_second_bert:
            predict_prob += \
                self.bert_scorer(query_id, predict_doc_id_highest, doc2index, "second") * 100

        return predict_prob


def get_recall_score(y_true, y_pred, th):
    base = list(zip(list(y_pred),list(y_true)))
    y_pred_sorted = sorted(base, key = lambda x : -x[0])
    if 1 in np.array(y_pred_sorted)[:th, 1]:
        return 1
    else:
        return 0

def get_mrr_score(y_true, y_pred):
    base = list(zip(list(y_pred),list(y_true)))
    y_pred_sorted = sorted(base, key = lambda x : -x[0])
    try:
        index = np.array(y_pred_sorted)[:,1].tolist().index(1)
        return 1/(index+1)
    except ValueError:
        return 0


parser = argparse.ArgumentParser()

#data
parser.add_argument("--id2doc_path", type=str, default=None, help="test data path")
parser.add_argument("--id2query_path", type=str, default=None, help="test data path")
parser.add_argument("--eval_query2doc_path", type=str, default=None, help="train data path")

parser.add_argument("--query_size", type=int, default=10000000000)
parser.add_argument("--faq_size", type=int, default=10000000000)

parser.add_argument("--exclude_negative_faq", action="store_true")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=4)


#model
parser.add_argument("--use_bert", action="store_true")
parser.add_argument("--model_name_or_path", nargs="*", type=str, default=[])
parser.add_argument("--bert_num_candidate", type=int, default=100)
parser.add_argument("--source_block_size", type=int, default=512)
parser.add_argument("--bert_task_type", type=str, default="classification")

#second_model
parser.add_argument("--use_second_bert", action="store_true")
parser.add_argument("--second_model_name_or_path", nargs="*", type=str, default=[])
parser.add_argument("--second_bert_num_candidate", type=int, default=10)
parser.add_argument("--second_source_block_size", type=int, default=512)
parser.add_argument("--second_bert_task_type", type=str, default="classification")

#bm25
parser.add_argument("--use_bm25", action="store_true")
parser.add_argument("--bm25_num_candidate", type=int, default=100)



args = parser.parse_args()

# Set seed
set_seed(args.seed)


# Get datasets
with open(args.id2doc_path) as f:
    id2doc = json.load(f)
with open(args.id2query_path) as f:
    id2query=json.load(f)
with open(args.eval_query2doc_path) as f:
    eval_query2doc=json.load(f)

all_doc_id = sorted(list(set(id2doc.keys())))

predictor=ScorePredictor(args, id2doc, id2query)

recall_1_list = []
recall_3_list = []
recall_10_list = []
recall_100_list = []
mrr_list = []
map_list = []
ndcg_10_list = []

true_dict = {}
pred_dict = {}

start_time = time.time()

for query_id, value in tqdm(eval_query2doc.items()):
    query_id = query_id
    positive_doc_id = value["positive_doc_id"]
    bm25_doc_id = value["bm25_doc_id"]

    if args.use_bm25:
        cand_doc_id = bm25_doc_id
    else:
        cand_doc_id = all_doc_id

    y_true = {doc_id:int(doc_id in positive_doc_id) for doc_id in cand_doc_id}
    y_pred = \
        {doc_id:score for doc_id, score in \
            zip(cand_doc_id, predictor.predict(query_id, cand_doc_id))}

    true_dict[query_id] = y_true
    pred_dict[query_id] = y_pred

end_time = time.time()

evaluator = pytrec_eval.RelevanceEvaluator(
    true_dict,
    {
        "recall.1,3,10,100",
        "ndcg_cut.10",
    })
score=evaluator.evaluate(pred_dict)

for query_id, score_dict in score.items():
    recall_1_list.append(score_dict[f"recall_1"])
    recall_3_list.append(score_dict[f"recall_3"])
    recall_10_list.append(score_dict[f"recall_10"])
    recall_100_list.append(score_dict[f"recall_100"])
    ndcg_10_list.append(score_dict[f"ndcg_cut_10"])

print(f"search size:{len(recall_1_list)}, search time:{(end_time-start_time)/len(recall_1_list)}")
print(f"recall@1: {np.average(recall_1_list)}")
print(f"recall@3: {np.average(recall_3_list)}")
print(f"recall@10: {np.average(recall_10_list)}")
print(f"recall@100: {np.average(recall_100_list)}")
print(f"NDCG@10: {np.average(ndcg_10_list)}")