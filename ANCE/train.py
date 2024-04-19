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
from paddle.nn import ClipGradByNorm
import faiss
from ANCE_model import ANCE, NLL_loss

parser = argparse.ArgumentParser(description='ANCE Training')
parser.add_argument('--dataset_path', type=str, default='/home/zhangyh/dataset/MS\ MARCO/passage\ raranking', help='Path to the training data dir')
parser.add_argument('--logdir', type=str, default='/home/zhangyh/code/paddle/logs')
parser.add_argument('--batchsize', type=int, default='2')
args = parser.parse_args()


def train(model, data_loader, optimizer, epochs=1, save_path='./checkpoints'):
    os.makedirs(args.logdir, exist_ok=True)
    writer = tbx.SummaryWriter(log_dir=args.logdir)
    model.train()
    
    index = faiss.IndexFlatIP(768)  
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    global_step = 0
    indexed_ids = set()  

    for epoch in range(epochs):
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit='batch') as pbar:
            for batch in data_loader:
                query_ids = batch['input_ids_query']
                pos_ids = batch['input_ids_pos']
                neg_ids = batch['input_ids_neg']

                query_mask = batch['attention_mask_query']
                pos_mask = batch['attention_mask_pos']
                neg_mask = batch['attention_mask_neg']

                batch_global_indices = batch['idx'] 
                global_to_batch_index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(batch_global_indices)}


                query_emb, pos_emb = model(query_ids, pos_ids, query_mask, pos_mask)

                if len(indexed_ids) > 5:  
                    
                    _, I = index.search(query_emb.numpy(), k=min(args.batchsize, 5))  
                    tmp_neg_ids = [int(idx) for idx in I[0]]  
                    valid_neg_ids = [global_to_batch_index_map[idx] for idx in tmp_neg_ids if idx in global_to_batch_index_map]

                    
                    if len(valid_neg_ids) > 0:
                        print(valid_neg_ids)
                        _, neg_emb = model(None, batch['input_ids_pos'][valid_neg_ids].unsqueeze(0), None, batch['attention_mask_pos'][valid_neg_ids].unsqueeze(0))
                        
                    else:
                         _, neg_emb = model(query_ids, neg_ids, query_mask, neg_mask)
                         
                    
                else:
                    
                    _, neg_emb = model(query_ids, neg_ids, query_mask, neg_mask)

                
                loss = NLL_loss(query_emb, pos_emb, neg_emb)
                
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                
                if epoch % 2 == 0:
                    #print("[!] begin updating ANN")
                    
                    for idx in range(len(batch['input_ids_query'])):
                        doc_id = batch['idx'][idx]
                        if doc_id not in indexed_ids:
                            
                            pos_text = batch['positive_text'][idx]
                            doc_tokens = tokenizer(pos_text, return_tensors='pd', padding='max_length', truncation=True, max_length=128,  return_attention_mask=True)
                            input_ids = doc_tokens['input_ids']
                            attention_mask = doc_tokens['attention_mask']

                            
                            _, doc_emb = model(None, input_ids, None, attention_mask)  
                            doc_emb_np = doc_emb.numpy()  
                            index.add(doc_emb_np)  
                            indexed_ids.add(doc_id)
                    #print("[!] update ANN")


                #print(loss.numpy())
                writer.add_scalar('loss', loss.numpy(), global_step)

               
                pbar.update(1)
                pbar.set_postfix({'loss': float(loss.numpy())})

                if global_step % 500 == 0 and global_step != 0:
                    paddle.save(model.state_dict(), f"{save_path}/ANCE_checkpoint_{global_step}.pdparams")
                    print(f"Checkpoint saved at step {global_step}")

                global_step += 1
    writer.close()  


print("[!] Load model")
model = ANCE()

# clip = ClipGradByNorm(clip_norm=1.0)  
# optimizer = AdamW(learning_rate=5e-5, parameters=model.parameters(), grad_clip=clip)

optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=5e-6)

collection_file = args.dataset_path + '/collection.tsv'
queries_file = args.dataset_path + '/queries.train.tsv'
qrels_file = args.dataset_path + '/qrels.train.tsv'

print("[!] Load dataset")
dataset = MS_MARCO_Dataset(queries_file=queries_file, qrels_file=qrels_file, collection_file=collection_file)
loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)


print("[!] Start training ")
train(model, loader, optimizer, epochs=1)
