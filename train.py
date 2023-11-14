
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
import math
import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import json

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

args = {}

class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        args,
        tokenizer,
        id2doc,
        id2query,
        query2doc,
        dataset_type = "train",
    ):

        self.tokenizer=tokenizer
        self.source_block_size = args.source_block_size
        self.task_type = args.task_type

        self.bos_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        self.eos_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        self.sep_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.pad_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.num_negative_sample = \
            args.train_num_negative_sample if dataset_type=="train" else \
            args.test_num_negative_sample
        self.negative_doc_cand_type = args.negative_doc_cand_type


        self.id2doc = id2doc
        self.id2query = id2query
        self.query2doc = {k:v for k,v in list(query2doc.items())[0:args.data_size]}

        self.doc_id_list = list(self.id2doc.keys())

        self.input_generate()

    # this func is called for every epoch
    def input_generate(self):

        self.epoch_examples = []

        for query_id, doc_id_items in self.query2doc.items():

            positive_doc_id_cand = doc_id_items["positive_doc_id"]
            if self.negative_doc_cand_type == "all":
                negative_doc_id_cand = self.doc_id_list
            elif self.negative_doc_cand_type == "bm25":
                negative_doc_id_cand = doc_id_items["bm25_doc_id"]
            elif self.negative_doc_cand_type == "all_bm25":
                if random.random()<0.5:
                    negative_doc_id_cand = self.doc_id_list
                else:
                    negative_doc_id_cand = doc_id_items["bm25_doc_id"]

            if len(negative_doc_id_cand)<self.num_negative_sample:
                continue

            #for one query, one positive answer and num_negative_sample answer are extracted
            positive_doc_id = random.sample(positive_doc_id_cand, 1)
            negative_doc_id = []

            #select negative_id for randomly, and if it is positive_id, continue the for loop.
            #this is continued until it extracts necessary negative_id.
            for _ in range(100):
                doc_id_cand = random.sample(negative_doc_id_cand, 1)[0]
                if doc_id_cand not in positive_doc_id_cand + negative_doc_id:
                    negative_doc_id.append(doc_id_cand)
                    if len(negative_doc_id)==self.num_negative_sample:
                        break
            if len(negative_doc_id)!=self.num_negative_sample:
                print("cannot find nengative_doc_id")
                print(positive_doc_id, negative_doc_id)
                exit()

            if self.task_type == "classification":
                pair = \
                    [(query_id, doc_id, 1) for doc_id in positive_doc_id] + \
                    [(query_id, doc_id, 0) for doc_id in negative_doc_id]

                for query_id, doc_id, label in pair:
                    text = \
                        self.id2query[query_id]["text_tokenized"] + \
                        [self.sep_token_id] + \
                        self.id2doc[doc_id]["text_tokenized"]
                    text = text[:self.source_block_size-2]
                    pad_size = self.source_block_size - 2 - len(text)

                    text = \
                        [self.bos_token_id] + \
                        text + \
                        [self.eos_token_id] + \
                        [self.pad_token_id] * pad_size

                    attention_mask = \
                        [1] * (self.source_block_size - pad_size) + \
                        [0] * pad_size

                    label = label

                    self.epoch_examples.append(
                        {
                            "text": text,
                            "attention_mask": attention_mask,
                            "label": label,
                        }
                    )

            elif self.task_type == "pairwise":
                pair = \
                    [(query_id, positive_doc_id[0], doc_id, 0) for doc_id in negative_doc_id] + \
                    [(query_id, doc_id, positive_doc_id[0], 1) for doc_id in negative_doc_id]

                for query_id, doc_id1, doc_id2, label in pair:

                    query_text_tokenized = self.id2query[query_id]["text_tokenized"]
                    doc_text_tokenized1 = self.id2doc[doc_id1]["text_tokenized"]
                    doc_text_tokenized2 = self.id2doc[doc_id2]["text_tokenized"]

                    length_size = \
                        len(query_text_tokenized) + \
                        len(doc_text_tokenized1) + \
                        len(doc_text_tokenized2) + \
                        1 + 1
                    over_size = length_size + 2 - self.source_block_size
                    if over_size>0:
                        doc_text_tokenized1 = doc_text_tokenized1[:-int((over_size+1)/2)]
                        doc_text_tokenized2 = doc_text_tokenized2[:-int((over_size+1)/2)]


                    text = \
                        query_text_tokenized + \
                        [self.sep_token_id] + \
                        doc_text_tokenized1 + \
                        [self.sep_token_id] + \
                        doc_text_tokenized2
                    text = text[:self.source_block_size-2]
                    pad_size = self.source_block_size - 2 - len(text)

                    text = \
                        [self.bos_token_id] + \
                        text + \
                        [self.eos_token_id] + \
                        [self.pad_token_id] * pad_size

                    attention_mask = \
                        [1] * (self.source_block_size - pad_size) + \
                        [0] * pad_size

                    label = label

                    self.epoch_examples.append(
                        {
                            "text": text,
                            "attention_mask": attention_mask,
                            "label": label,
                        }
                    )

    def __len__(self):
        return len(self.epoch_examples)

    def __getitem__(self, i) -> torch.Tensor:

        return {k:torch.tensor(v, dtype=torch.long) for k,v in self.epoch_examples[i].items()}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--id2doc_path", type=str, default=None, help="test data path")
    parser.add_argument("--id2query_path", type=str, default=None, help="test data path")

    parser.add_argument("--train_query2doc_path", type=str, default=None, help="train data path")
    parser.add_argument("--eval_query2doc_path", type=str, default=None, help="train data path")
    parser.add_argument("--test_query2doc_path", type=str, default=None, help="train data path")

    parser.add_argument("--source_block_size", type=int, default=512, help="max sentence size")
    parser.add_argument("--target_block_size", type=int, default=128, help="max sentence size")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank of logger")

    parser.add_argument("--output_dir", type=str, default="./model/fine_tuned_models/tmp", help="output dir")
    parser.add_argument("--do_train", action="store_true", help="do training")
    parser.add_argument("--do_eval", action="store_true", help="do evaluation")
    parser.add_argument("--do_generate", action="store_true", help="do generation of the sample texts")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="eval and batch size")
    parser.add_argument("--per_device_generate_batch_size", type=int, default=8, help="generation batch size")
    parser.add_argument("--total_batch_size", type=int, default=256, help="train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="graddient accumulation steps. automatically decided")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="optimizer params")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="optimizer params")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="optimizer params")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="clip large gradient")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="epochs")
    parser.add_argument("--eval_freq", type=int, default=1, help="eval frequent")

    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--data_seed", type=int, default=None, help="data seed")
    parser.add_argument("--n_gpu", type=int, default=1, help="gpu num")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--ignore_index", type=int, default=-100,
    help="ignore index of the crossentropyloss")

    #data config
    parser.add_argument("--data_size", type=int, default=100000000000000, help="data size")
    parser.add_argument("--train_num_negative_sample", type=int, default=1)
    parser.add_argument("--test_num_negative_sample", type=int, default=4)
    parser.add_argument("--negative_doc_cand_type", type=str, default="all")

    #model config
    parser.add_argument("--model_name_or_path", type=str, default=None, help="model name or path")
    parser.add_argument("--config_name", type=str, default=None, help="config name")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="tokenizer name. if not specified, use tokenizer depending on model")

    #training config
    parser.add_argument("--label_smoothing", type=float, default=0.0,
    help="label smoothing param for loss function")
    parser.add_argument("--task_type", type=str, default="classification", help="task_type")
    parser.add_argument("--num_labels", type=int, default=2, help="classification num labels")

    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    #parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    #args, args, args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    logger.info("Training/evaluation parameters %s", args)

    # Set seed for model initialization (fixed to 0)
    set_seed(args.seed)

    #config
    print(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=None)
    config.num_labels = args.num_labels

    #tokenizer
    tokenizer = \
        AutoTokenizer.from_pretrained(args.tokenizer_name_or_path) \
            if args.tokenizer_name_or_path != None else \
        AutoTokenizer.from_pretrained(args.model_name_or_path)

    #model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model=model.to(args.device)

    # Set seed for data order or other
    if args.data_seed!=None:
        set_seed(args.data_seed)

    #batch size
    if torch.cuda.device_count()>1:
        model=torch.nn.DataParallel(model)
    args.per_device_train_batch_size*=torch.cuda.device_count()
    args.per_device_eval_batch_size*=torch.cuda.device_count()
    args.gradient_accumulation_steps=int(args.total_batch_size/args.per_device_train_batch_size)


    logger.info(f"train batch size: {args.per_device_train_batch_size}, \
        gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    # Get datasets
    with open(args.id2doc_path) as f:
        id2doc = json.load(f)
    with open(args.id2query_path) as f:
        id2query=json.load(f)
    with open(args.train_query2doc_path) as f:
        train_query2doc=json.load(f)
    with open(args.eval_query2doc_path) as f:
        eval_query2doc=json.load(f)

    train_dataset = \
        TextDataset(
            args,
            tokenizer=tokenizer,
            id2doc=id2doc,
            id2query=id2query,
            query2doc = train_query2doc,
            dataset_type="train") \
        if args.do_train else []
    eval_dataset = \
        TextDataset(
            args,
            tokenizer=tokenizer,
            id2doc=id2doc,
            id2query=id2query,
            query2doc = eval_query2doc,
            dataset_type="eval")

    logger.info("train_dataset_size: {}, eval_dataset_size: {}"
        .format(len(train_dataset), len(eval_dataset)))

    # DataLoader
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=None,
        batch_size=args.per_device_eval_batch_size)

    # Optimizer, Scheduler
    total_steps= \
            int(math.ceil(len(train_dataset)/ \
            (args.per_device_train_batch_size*args.gradient_accumulation_steps))* \
            args.num_train_epochs) if args.do_train else 0

    warmup_steps=math.ceil(total_steps*0.06)
    logger.info(f"total steps: {total_steps}, warmup_steps: {warmup_steps}")

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() \
                if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr":args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() \
                if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr":args.learning_rate,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fct=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    os.makedirs(args.output_dir, exist_ok=True)

    def train():
        progress_bar = tqdm(range(total_steps))
        for epoch in range(args.num_train_epochs):
            logger.info("start epoch {}".format(epoch))
            train_dataset.input_generate()
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, collate_fn=None,
                batch_size=args.per_device_train_batch_size)

            model.train()
            for step, batch in enumerate(train_dataloader):

                text = batch["text"].to(args.device)
                label = batch["label"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)

                outputs=model(input_ids=text, attention_mask = attention_mask)[0] #(batch_size)

                loss=loss_fct(outputs, label)

                if torch.cuda.device_count()>1:
                    loss = loss.mean()
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
            if (epoch+1)%args.eval_freq==0:
                evaluate(epoch)

    def evaluate(epoch=None):
        model.eval()

        losses = []
        preds = []
        labels = []
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            with torch.no_grad():
                text = batch["text"].to(args.device)
                label = batch["label"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                outputs=model(input_ids=text, attention_mask = attention_mask)[0]

            losses.append(loss_fct(outputs, label).item())
            preds += outputs.argmax(dim=-1).cpu().numpy().tolist()
            labels += label.cpu().numpy().tolist()

        loss = np.average(losses)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        accuracy = accuracy_score(labels, preds)
        result = {
            "precision":precision, "recall":recall, "f1": f1,
            "accuracy": accuracy, "loss": loss
        }

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            writer.write("***** Eval results: Epoch {} *****\n".format(epoch))
            for key in sorted(result.keys()):
                logger.info("  {} = {}".format(key, str(result[key])))
                writer.write("{} = {}\n".format(key, str(result[key])))

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        train()

    # Training
    if args.do_eval:
        logger.info("*** Eval ***")
        evaluate()

    #save the model
    if args.do_train and args.output_dir is not None:
        if torch.cuda.device_count()>1:
            model=model.module
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
