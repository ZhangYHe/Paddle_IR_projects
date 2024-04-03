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
args = parser.parse_args()


file_path = args.file_path

file_path = '/home/zhangyh/code/paddle/DPR_paddle/data/SQuAD2.0/train-v2.0.json'
squad_examples = load_squad_data(file_path)

# print(squad_examples[:3])

processed_data = preprocess_data_for_dpr(squad_examples)
print(processed_data[10])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SquadDPRDataset(processed_data, tokenizer)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)


model = DPR()
optimizer = AdamW(learning_rate=5e-5, parameters=model.parameters())
checkpoint_path = "./checkpoints/"

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, batch in progress_bar:
        question_encoding, positive_encoding, negative_encoding = batch
        
        question_embeddings = model.question_encoder(**question_encoding)
        positive_embeddings = model.context_encoder(**positive_encoding)
        negative_embeddings = model.context_encoder(**negative_encoding)

        positive_similarity = compute_similarity(question_embeddings, positive_embeddings)
        negative_similarity = compute_similarity(question_embeddings, negative_embeddings)
        
        loss = F.relu(negative_similarity - positive_similarity + 0.5).mean()
        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        total_loss += loss.numpy()  


        if step % 10 == 0:
            avg_loss = total_loss / (step + 1)
            progress_bar.set_description(f'Epoch {epoch+1} Step {step} Avg Loss: {avg_loss:.4f}')
    

    avg_loss = total_loss / len(data_loader)
    print(f'End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    

    checkpoint_filename = f'dpr_model_epoch_{epoch}.pdparams'
    checkpoint_filepath = checkpoint_path + checkpoint_filename
    paddle.save(model.state_dict(), checkpoint_filepath)
    print(f'Checkpoint saved to {checkpoint_filepath}')
