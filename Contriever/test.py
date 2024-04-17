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

parser = argparse.ArgumentParser(description='Contriever Testing')
parser.add_argument('--dataset_path', type=str, default='/home/zhangyh/dataset/MS\ MARCO/passage\ raranking', help='Path to the training data dir')
parser.add_argument('--checkpoint_path', type=str, default='/home/zhangyh/code/paddle/Contriever_paddle/checkpoints/Contriever_model_checkpoint_4000.pdparams')
args = parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    state_dict = paddle.load(checkpoint_path)
    model.set_state_dict(state_dict)

def evaluate_accuracy(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with paddle.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids_query = batch['input_ids_query']
            attention_mask_query = batch['attention_mask_query']
            input_ids_pos = batch['input_ids_pos']
            attention_mask_pos = batch['attention_mask_pos']

            
            query_embeddings = model(input_ids_query, attention_mask=attention_mask_query)
            positive_embeddings = model(input_ids_pos, attention_mask=attention_mask_pos)

            
            similarities = F.cosine_similarity(query_embeddings, positive_embeddings)

            
            predicted_correct = similarities > 0.6  
            correct += paddle.to_tensor(predicted_correct).numpy().sum()
            total += len(predicted_correct)
            print(f"current accuracy : {correct / total}")

    accuracy = correct / total
    return accuracy

print("[!] Load model")
model = ContrieverModel()
checkpoint_path = args.checkpoint_path
load_checkpoint(model, checkpoint_path)


print("[!] Load dataset")
collection_file = args.dataset_path + '/collection.tsv'
queries_file = args.dataset_path + '/queries.dev.tsv'
qrels_file = args.dataset_path + '/qrels.dev.tsv'


test_dataset = MS_MARCO_Dataset(queries_file=queries_file, qrels_file=qrels_file, collection_file=collection_file)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


print("[!] Evaluate")
accuracy = evaluate_accuracy(model, test_loader)
print(f"Accuracy: {accuracy}")