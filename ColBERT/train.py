import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from ColBERT_model import ColBERT, margin_ranking_loss
from dataset import MS_MARCO_Dataset
import os
from paddle.optimizer import AdamW
from paddle.regularizer import L2Decay
from paddlenlp.transformers import BertModel, BertTokenizer
import argparse
import time

parser = argparse.ArgumentParser(description='ColBERT Training')
parser.add_argument('--dataset_path', type=str, default='/home/zhangyh/dataset/MS\ MARCO/passage\ raranking', help='Path to the training data dir')
args = parser.parse_args()


def train(model, data_loader, optimizer, epochs=1, save_path='./checkpoints'):
    model.train()
    global_step = 0
    for epoch in range(epochs):
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit='batch') as pbar:
            for batch in data_loader:
                input_ids_query, attention_mask_query = batch['input_ids_query'], batch['attention_mask_query']
                input_ids_pos, attention_mask_pos = batch['input_ids_pos'], batch['attention_mask_pos']
                input_ids_neg, attention_mask_neg = batch['input_ids_neg'], batch['attention_mask_neg']

                # start_time = time.time()
                
                query_repr, pos_repr = model(input_ids_query, attention_mask_query, input_ids_pos, attention_mask_pos)
                _, neg_repr = model(input_ids_query, attention_mask_query, input_ids_neg, attention_mask_neg)
                # print(f"model time : {time.time()-start_time}")
                # start_time = time.time()

                
                pos_scores = model.score(query_repr, pos_repr)
                neg_scores = model.score(query_repr, neg_repr)

                
                loss = margin_ranking_loss(model, pos_scores, neg_scores, margin=0.1)
                # print(f"loss time: {time.time()-start_time}")
                
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                
                pbar.update(1)
                pbar.set_postfix({'loss': float(loss.numpy())})


                
                if global_step % 500 == 0 and global_step != 0:
                    paddle.save(model.state_dict(), f"{save_path}/colbert_checkpoint_{global_step}.pdparams")
                    print(f"Checkpoint saved at step {global_step}")

                global_step += 1


print("[!] Load model")
model = ColBERT()
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=5e-5)

collection_file = args.dataset_path + '/collection.tsv'
queries_file = args.dataset_path + '/queries.train.tsv'
qrels_file = args.dataset_path + '/qrels.train.tsv'

print("[!] Load dataset")
dataset = MS_MARCO_Dataset(queries_file=queries_file, qrels_file=qrels_file, collection_file=collection_file)
loader = DataLoader(dataset, batch_size=40, shuffle=True)

save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)


print("[!] Start training ")
train(model, loader, optimizer, epochs=3)
