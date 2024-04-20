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
from ANCE_model import ANCE, NLL_loss

parser = argparse.ArgumentParser(description='ANCE Testing')
parser.add_argument('--dataset_path', type=str, default='/home/zhangyh/dataset/MS\ MARCO/passage\ raranking', help='Path to the training data dir')
parser.add_argument('--checkpoint_path', type=str, default='/home/zhangyh/code/paddle/ColBERT_paddle/checkpoints/colbert_checkpoint_1500.pdparams')
parser.add_argument('--logdir', type=str, default='/home/zhangyh/code/paddle/logs')
args = parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    state_dict = paddle.load(checkpoint_path)
    model.set_state_dict(state_dict)

def test(model, data_loader):
    model.eval()

    writer = tbx.SummaryWriter(log_dir=args.logdir)
    total_accuracy = 0
    total_samples = 0
    total_count = 0
    correct_count = 0

    with tqdm(total=len(data_loader), desc="Testing", unit='batch') as pbar:
        for i, batch in enumerate(data_loader):
            query_ids = batch['input_ids_query']
            pos_ids = batch['input_ids_pos']
            neg_ids = batch['input_ids_neg']

            query_mask = batch['attention_mask_query']
            pos_mask = batch['attention_mask_pos']
            neg_mask = batch['attention_mask_neg']

            # Compute embeddings
            query_emb, pos_emb = model(query_ids, pos_ids, query_mask, pos_mask)
            _, neg_emb = model(query_ids, neg_ids, query_mask, neg_mask)

            # Compute similarity scores
            pos_score = paddle.sum(query_emb * pos_emb, axis=1)
            neg_score = paddle.sum(query_emb * neg_emb, axis=1)

            
            correct = (pos_score > neg_score).astype(paddle.int32).sum().item()
            correct_count += correct

            total_count += query_ids.shape[0]

            pbar.set_postfix({'Batch Accuracy': float(correct/query_ids.shape[0])})
            pbar.update(1)
            writer.add_scalar('Batch Accuracy', float(correct/query_ids.shape[0]), i)
            writer.add_scalar('Total Accuracy', correct_count / total_count, i)
            print('Total Accuracy', correct_count / total_count)

    
    overall_accuracy = correct_count / total_count
    print(f"Overall Accuracy: {overall_accuracy}")
    writer.add_scalar('accuracy', overall_accuracy, 0)
    writer.close()

def accuracy(s_pos, s_neg):
    #s_pos = s_pos.unsqueeze(-1)
    # print(s_pos.shape)
    # print(s_neg.shape)
    corrects = (s_pos > s_neg).astype(paddle.float32).mean()
    return corrects


print("[!] Load model")
model = ANCE()
checkpoint_path = args.checkpoint_path
load_checkpoint(model, checkpoint_path)


print("[!] Load dataset")
collection_file = args.dataset_path + '/collection.tsv'
queries_file = args.dataset_path + '/queries.dev.tsv'
qrels_file = args.dataset_path + '/qrels.dev.tsv'


test_dataset = MS_MARCO_Dataset(queries_file=queries_file, qrels_file=qrels_file, collection_file=collection_file)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)


print("[!] Evaluate")
test(model, test_loader)