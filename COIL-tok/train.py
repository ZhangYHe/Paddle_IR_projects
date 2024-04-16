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
import time
import tensorboardX as tbx
from COIL_model import COIL_tok_Model, coil_loss

parser = argparse.ArgumentParser(description='COIL-tok Training')
parser.add_argument('--dataset_path', type=str, default='/home/zhangyh/dataset/MS\ MARCO/passage\ raranking', help='Path to the training data dir')
parser.add_argument('--logdir', type=str, default='/home/zhangyh/code/paddle/logs')
args = parser.parse_args()


def train(model, data_loader, optimizer, epochs=1, save_path='./checkpoints'):
    os.makedirs(args.logdir, exist_ok=True)
    writer = tbx.SummaryWriter(log_dir=args.logdir) 
    model.train()
    global_step = 0
    for epoch in range(epochs):
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit='batch') as pbar:
            for batch in data_loader:
                input_ids_query, attention_mask_query = batch['input_ids_query'], batch['attention_mask_query']
                input_ids_pos, attention_mask_pos = batch['input_ids_pos'], batch['attention_mask_pos']
                input_ids_neg, attention_mask_neg = batch['input_ids_neg'], batch['attention_mask_neg']

                
                s_pos = model(input_ids_query, attention_mask_query, input_ids_pos, attention_mask_pos)
               
                s_neg = model(input_ids_query, attention_mask_query, input_ids_neg, attention_mask_neg)

              
                loss = coil_loss(s_pos, s_neg)

             
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

               
                writer.add_scalar('loss', loss.numpy(), global_step)

               
                pbar.update(1)
                pbar.set_postfix({'loss': float(loss.numpy())})


               
                if global_step % 500 == 0 and global_step != 0:
                    paddle.save(model.state_dict(), f"{save_path}/COIL-tok_checkpoint_{global_step}.pdparams")
                    print(f"Checkpoint saved at step {global_step}")

                global_step += 1
    writer.close()  


print("[!] Load model")
model = COIL_tok_Model(model_name="COIL-tok")
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=5e-6)

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
