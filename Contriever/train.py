import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from Contriever_model import ContrieverModel, contrastive_loss
from dataset import MS_MARCO_Dataset
import os
from paddle.optimizer import AdamW
from paddle.regularizer import L2Decay
from paddlenlp.transformers import BertModel, BertTokenizer
import argparse


parser = argparse.ArgumentParser(description='Contriever Training')
parser.add_argument('--dataset_path', type=str, default='/home/zhangyh/dataset/MS\ MARCO/passage\ raranking', help='Path to the training data dir')
args = parser.parse_args()


def train(model, data_loader, optimizer, save_dir, epochs=3, tau=0.05):
    global_step = 0
    for epoch in range(epochs):
        model.train()
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit='batch') as pbar:
            for batch in data_loader:
               
                input_ids_query = batch['input_ids_query']
                attention_mask_query = batch['attention_mask_query']
                input_ids_pos = batch['input_ids_pos']
                attention_mask_pos = batch['attention_mask_pos']
                input_ids_neg = batch['input_ids_neg']
                attention_mask_neg = batch['attention_mask_neg']

               
                query_embeddings = model(input_ids_query, attention_mask=attention_mask_query)
                positive_embeddings = model(input_ids_pos, attention_mask=attention_mask_pos)
                negative_embeddings = model(input_ids_neg, attention_mask=attention_mask_neg)
                
                
                loss = contrastive_loss(query_embeddings, positive_embeddings, negative_embeddings, tau)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                
                pbar.set_postfix(loss=loss.numpy())
                pbar.update(1)
                
                if global_step % 500 == 0:  
                    paddle.save(model.state_dict(), os.path.join(save_dir, f'Contriever_model_checkpoint_{global_step}.pdparams'))
                global_step += 1

        
        paddle.save(model.state_dict(), os.path.join(save_dir, f'Contriever_model_checkpoint_epoch_{epoch+1}.pdparams'))


print("[!] Load model")
model = ContrieverModel()
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=5e-5)

collection_file = args.dataset_path + '/collection.tsv'
queries_file = args.dataset_path + '/queries.train.tsv'
qrels_file = args.dataset_path + '/qrels.train.tsv'

print("[!] Load dataset")
dataset = MS_MARCO_Dataset(queries_file=queries_file, qrels_file=qrels_file, collection_file=collection_file)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)


print("[!] Start training ")
train(model, loader, optimizer, save_dir, epochs=3)
