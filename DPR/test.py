import json
import random
from paddle.io import Dataset, DataLoader
import numpy as np
import paddle
from paddle.optimizer import AdamW
from dataset import SquadDPRDataset, load_squad_data, preprocess_data_for_dpr
from DPR_model import DPR,compute_similarity
from paddlenlp.transformers import BertModel, BertTokenizer
import paddle.nn.functional as F
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='DPR Training')
parser.add_argument('--file_path', type=str, default='/home/zhangyh/code/paddle/DPR_paddle/data/SQuAD2.0/train-v2.0.json', help='Path to the training data file')
parser.add_argument('--checkpoint_path', type=str, default='/home/zhangyh/code/paddle/DPR_paddle/checkpoints/checkpointsdpr_model_epoch_2.pdparams')
args = parser.parse_args()

model = DPR()
checkpoint_path = args.checkpoint_path
model_state_dict = paddle.load(checkpoint_path)
model.set_state_dict(model_state_dict)

file_path = args.file_path
squad_examples = load_squad_data(file_path)
processed_data = preprocess_data_for_dpr(squad_examples)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SquadDPRDataset(processed_data, tokenizer)
data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

# inference
model.eval()
correct = 0
total = 0
for batch in tqdm(data_loader):
    question_encoding, positive_encoding, negative_encoding = batch

    question_embeddings = model.question_encoder(**question_encoding)
    positive_embeddings = model.context_encoder(**positive_encoding)
    negative_embeddings = model.context_encoder(**negative_encoding)

    positive_similarity = compute_similarity(question_embeddings, positive_embeddings)
    negative_similarity = compute_similarity(question_embeddings, negative_embeddings)

    is_positive_greater = paddle.greater_than(positive_similarity, negative_similarity)

    is_positive_greater = paddle.mean(is_positive_greater)

    question_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(question_encoding['input_ids'][0].numpy()))
    positive_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(positive_encoding['input_ids'][0].numpy()))
    negative_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(negative_encoding['input_ids'][0].numpy()))

    print("Question: ", question_text)
    print("Answer: ", positive_text if is_positive_greater.numpy() else negative_text)