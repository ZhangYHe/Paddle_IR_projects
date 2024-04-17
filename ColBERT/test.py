import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset import MS_MARCO_Dataset
import os
from paddle.optimizer import AdamW
from paddle.regularizer import L2Decay
from paddlenlp.transformers import BertModel, BertTokenizer
import argparse
from ColBERT_model import ColBERT, margin_ranking_loss

parser = argparse.ArgumentParser(description='ColBERT Testing')
parser.add_argument('--dataset_path', type=str, default='/home/zhangyh/dataset/MS\ MARCO/passage\ raranking', help='Path to the training data dir')
parser.add_argument('--checkpoint_path', type=str, default='/home/zhangyh/code/paddle/ColBERT_paddle/checkpoints/colbert_checkpoint_1500.pdparams')
args = parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    state_dict = paddle.load(checkpoint_path)
    model.set_state_dict(state_dict)

def test(model, data_loader):
    model.eval()  
    correct = 0
    total = 0

    with paddle.no_grad():  
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids_query, attention_mask_query = batch['input_ids_query'], batch['attention_mask_query']
            input_ids_pos, attention_mask_pos = batch['input_ids_pos'], batch['attention_mask_pos']
            input_ids_neg, attention_mask_neg = batch['input_ids_neg'], batch['attention_mask_neg']

            
            query_repr, pos_repr = model(input_ids_query, attention_mask_query, input_ids_pos, attention_mask_pos)
            _, neg_repr = model(input_ids_query, attention_mask_query, input_ids_neg, attention_mask_neg)

            
            pos_scores = model.score(query_repr, pos_repr)
            neg_scores = model.score(query_repr, neg_repr)

            
            correct += paddle.sum(pos_scores > neg_scores).numpy()
            total += len(pos_scores)
            print(f"Accuracy: {correct / total:.4f}")

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

print("[!] Load model")
model = ColBERT()
checkpoint_path = args.checkpoint_path
load_checkpoint(model, checkpoint_path)


print("[!] Load dataset")
collection_file = args.dataset_path + '/collection.tsv'
queries_file = args.dataset_path + '/queries.dev.tsv'
qrels_file = args.dataset_path + '/qrels.dev.tsv'


test_dataset = MS_MARCO_Dataset(queries_file=queries_file, qrels_file=qrels_file, collection_file=collection_file)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True)


print("[!] Evaluate")
test(model, test_loader)