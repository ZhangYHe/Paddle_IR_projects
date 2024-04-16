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

parser = argparse.ArgumentParser(description='ColBERT Training')
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

    with tqdm(total=len(data_loader), desc="Testing", unit='batch') as pbar:
        for i, batch in enumerate(data_loader):
            input_ids_query, attention_mask_query = batch['input_ids_query'], batch['attention_mask_query']
            input_ids_pos, attention_mask_pos = batch['input_ids_pos'], batch['attention_mask_pos']
            input_ids_neg, attention_mask_neg = batch['input_ids_neg'], batch['attention_mask_neg']

            
            s_pos = model(input_ids_query, attention_mask_query, input_ids_pos, attention_mask_pos)
            s_neg = model(input_ids_query, attention_mask_query, input_ids_neg, attention_mask_neg)

            
            acc = accuracy(s_pos, s_neg)
            total_accuracy += acc.numpy() * len(input_ids_query)
            total_samples += len(input_ids_query)

            pbar.set_postfix({'Batch Accuracy': float(acc.numpy())})
            pbar.update(1)
            writer.add_scalar('Batch Accuracy', acc.numpy(), i)
            writer.add_scalar('Total Accuracy', total_accuracy / total_samples, i)
            print('Total Accuracy', total_accuracy / total_samples)

    
    overall_accuracy = total_accuracy / total_samples
    print(f"Overall Accuracy: {overall_accuracy}")
    writer.add_scalar('accuracy', overall_accuracy, 0)
    writer.close()

def accuracy(s_pos, s_neg):
    s_pos = s_pos.unsqueeze(-1)
    corrects = (s_pos > s_neg).astype(paddle.float32).mean()
    return corrects


print("[!] Load model")
model = COIL_tok_Model(model_name="COIL-tok")
checkpoint_path = args.checkpoint_path
load_checkpoint(model, checkpoint_path)


print("[!] Load dataset")
collection_file = args.dataset_path + '/collection.tsv'
queries_file = args.dataset_path + '/queries.dev.tsv'
qrels_file = args.dataset_path + '/qrels.dev.tsv'


test_dataset = MS_MARCO_Dataset(queries_file=queries_file, qrels_file=qrels_file, collection_file=collection_file)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


print("[!] Evaluate")
test(model, test_loader)