from paddle.io import Dataset, DataLoader
import paddle
from paddlenlp.transformers import BertTokenizer
import json
import random

class SquadDPRDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        positive_context = item['positive_context']
        negative_context = item['negative_context']

        question_encoding = self.tokenizer(question, max_length=self.max_length, truncation=True, padding='max_length', return_attention_mask=True)
        positive_encoding = self.tokenizer(positive_context, max_length=self.max_length, truncation=True, padding='max_length', return_attention_mask=True)
        negative_encoding = self.tokenizer(negative_context, max_length=self.max_length, truncation=True, padding='max_length', return_attention_mask=True)

 
        question_encoding = {k: paddle.to_tensor(v) for k, v in question_encoding.items()}
        positive_encoding = {k: paddle.to_tensor(v) for k, v in positive_encoding.items()}
        negative_encoding = {k: paddle.to_tensor(v) for k, v in negative_encoding.items()}

        return question_encoding, positive_encoding, negative_encoding

def load_squad_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    data = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                id = qa['id']
                is_impossible = qa.get('is_impossible', False)
                answers = [answer['text'] for answer in qa['answers']]
                
                
                data.append({
                    'id': id,
                    'question': question,
                    'context': context,
                    'answers': answers,
                    'is_impossible': is_impossible
                })
    return data

def preprocess_data_for_dpr(squad_data):
    processed_data = []
    for item in squad_data:
        if item['is_impossible']:
            continue  
        question = item['question']
        context = item['context']
        answer = item['answers'][0]  
        
        
        positive_context = context
        
        
        negative_context = context[::-1]  
        
        processed_data.append({
            'question': question,
            'positive_context': positive_context,
            'negative_context': negative_context
        })
    return processed_data