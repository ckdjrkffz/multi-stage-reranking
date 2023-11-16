# Text Retrieval with Multi-Stage Re-Ranking Models
- This is the code for the paper "Text Retrieval with Multi-Stage Re-Ranking Models". [arXiv](https://arxiv.org/pdf/2311.07994.pdf)


## Requirment

- Python library
  - `pip install requirements.txt`
- Java
  - For BM25 search, we use pyserini library, and it requires Java 11. Use following command or other methods.
    - `sudo apt update; sudo apt install openjdk-11-jdk`

## Dataset

### Download

- We use the BEIR dataset.
- You can download from [here](https://github.com/beir-cellar/beir) or use the following script.

```
source download_dataset.sh msmarco
source download_dataset.sh fiqa
source download_dataset.sh scifact
source download_dataset.sh hotpotqa
```

### Preprocess

- We use the BEIR dataset.
- You can preprocess the dataset by the following script.
  - Tokenizing the dataset and the text search by BM25 are done in the preprocess.
```
source preprocess_dataset.sh msmarco
source preprocess_dataset.sh fiqa
source preprocess_dataset.sh scifact
source preprocess_dataset.sh hotpotqa
```

## Model

- We fine-tune the following pre-trained models:
  - 1. MiniLM-L6-H384-distilled-from-RoBERTa-Large
  - 2. MiniLM-L6-H768-distilled-from-RoBERTa-Large
- You can download the models from [here](https://github.com/microsoft/unilm/tree/master/minilm).
  - The vocab and tokenizer are not included in this model. The vocab and the tokenizer are from RoBERTa-Large, so you can download from [here](https://huggingface.co/roberta-large).
  - Download these models and place them in the `model/pre_trained_models`
- Alternatively, you can use huggingface version. H384 model is [here](https://huggingface.co/nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large) and H768(https://huggingface.co/nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large) is [here].
  - These models should be identical to the models of the github version.
  - If you use these models, replace the some path in the code.


## Training

### Normal (pointwise) LM

```
python train.py \
--model_name_or_path ./model/pre_trained_models/MiniLM-L6-H384-distilled-from-RoBERTa-Large \
--tokenizer_name_or_path ./model/pre_trained_models/roberta-large \
--do_train \
--task_type classification --negative_doc_cand_type all \
--id2doc_path dataset/beir/processed/msmarco/document.json \
--id2query_path dataset/beir/processed/msmarco/query.json \
--train_query2doc_path dataset/beir/processed_bm25/msmarco/qrels/train.json \
--eval_query2doc_path dataset/beir/processed_bm25/msmarco/qrels/dev.json \
--output_dir ./model/fine_tuned_models/MiniLM_L6_H384_msmarco_classification_all_e10_ns1_lr5e-5_s0 \
--num_train_epochs 10 --learning_rate 5e-5 --seed 0 \
--per_device_train_batch_size 16 --per_device_eval_batch_size 16 --total_batch_size 64 \
--source_block_size 512
```

### Pairwise LM

```
python train.py \
--model_name_or_path ./model/pre_trained_models/MiniLM-L6-H384-distilled-from-RoBERTa-Large \
--tokenizer_name_or_path ./model/pre_trained_models/roberta-large \
--do_train \
--task_type pairwise --negative_doc_cand_type all \
--id2doc_path dataset/beir/processed/msmarco/document.json \
--id2query_path dataset/beir/processed/msmarco/query.json \
--train_query2doc_path dataset/beir/processed_bm25/msmarco/qrels/train.json \
--eval_query2doc_path dataset/beir/processed_bm25/msmarco/qrels/dev.json \
--output_dir ./model/fine_tuned_models/MiniLM_L6_H384_msmarco_pairwise_all_e30_ns1_lr5e-5_s0 \
--num_train_epochs 30 --learning_rate 5e-5 --seed 0 \
--per_device_train_batch_size 16 --per_device_eval_batch_size 16 --total_batch_size 64 \
--source_block_size 512
```


## Evaluation

### only BM25

```
python -u evaluate.py \
--id2doc_path dataset/beir/processed/fiqa/document.json \
--id2query_path dataset/beir/processed/fiqa/query.json \
--eval_query2doc_path dataset/beir/processed_bm25/fiqa/qrels/test.json \
--use_bm25
```


### BM25 + Normal LM

```
python -u evaluate.py \
--id2doc_path dataset/beir/processed/fiqa/document.json \
--id2query_path dataset/beir/processed/fiqa/query.json \
--eval_query2doc_path dataset/beir/processed_bm25/fiqa/qrels/test.json \
--batch_size 16 \
--bert_num_candidate 100 \
--source_block_size 512 \
--bert_task_type classification \
--use_bm25 --use_bert \
--model_name_or_path \
./model/fine_tuned_models/MiniLM_L6_H384_msmarco_classification_all_e10_ns1_lr5e-5_s0
```

### BM25 + Normal LM + Ensemble

- Prepare multiple models, e.g., by training models with multiple seeds.

```
python -u evaluate.py \
--id2doc_path dataset/beir/processed/fiqa/document.json \
--id2query_path dataset/beir/processed/fiqa/query.json \
--eval_query2doc_path dataset/beir/processed_bm25/fiqa/qrels/test.json \
--batch_size 16 \
--bert_num_candidate 100 --second_bert_num_candidate 10 \
--source_block_size 512 --second_source_block_size 512 \
--bert_task_type classification --second_bert_task_type classification \
--use_bm25 --use_bert --use_second_bert \
--model_name_or_path \
./model/fine_tuned_models/MiniLM_L6_H384_msmarco_classification_all_e10_ns1_lr5e-5_s0 \
--second_model_name_or_path \
./model/fine_tuned_models/MiniLM_L6_H384_msmarco_classification_all_e10_ns1_lr5e-5_s0 \
./model/fine_tuned_models/MiniLM_L6_H384_msmarco_classification_all_e10_ns1_lr5e-5_s1 \
./model/fine_tuned_models/MiniLM_L6_H384_msmarco_classification_all_e10_ns1_lr5e-5_s2
```

### BM25 + Normal LM + Pairwise LM

```
python -u evaluate.py \
--id2doc_path dataset/beir/processed/fiqa/document.json \
--id2query_path dataset/beir/processed/fiqa/query.json \
--eval_query2doc_path dataset/beir/processed_bm25/fiqa/qrels/test.json \
--batch_size 16 \
--bert_num_candidate 100 --second_bert_num_candidate 10 \
--source_block_size 512 --second_source_block_size 512 \
--bert_task_type classification --second_bert_task_type pairwise \
--use_bm25 --use_bert --use_second_bert \
--model_name_or_path \
./model/fine_tuned_models/MiniLM_L6_H384_msmarco_classification_all_e10_ns1_lr5e-5_s0 \
--second_model_name_or_path \
./model/fine_tuned_models/MiniLM_L6_H384_msmarco_pairwise_all_e30_ns1_lr5e-5_s0
```


## LICENSE

Apache 2.0