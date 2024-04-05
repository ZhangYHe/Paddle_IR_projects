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

class ContrieverModel(nn.Layer):
    def __init__(self):
        super(ContrieverModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):

        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                   token_type_ids=token_type_ids,
                                                   position_ids=position_ids,
                                                   attention_mask=attention_mask)
        return pooled_output

def prepare_data(text, tokenizer, max_seq_len=128):
    
    encoded_text = tokenizer(text=text, max_seq_len=max_seq_len, pad_to_max_seq_len=True, return_attention_mask=True)
    input_ids = np.array(encoded_text["input_ids"], dtype="int64")
    token_type_ids = np.array(encoded_text["token_type_ids"], dtype="int64")
    attention_mask = np.array(encoded_text["attention_mask"], dtype="int64")
    return input_ids, token_type_ids, attention_mask

def contrastive_loss(query_embeddings, positive_embeddings, negative_embeddings, tau=0.05):
    
    pos_scores = F.cosine_similarity(query_embeddings, positive_embeddings, axis=-1) / tau
    pos_scores = pos_scores.unsqueeze(1)  

    
    all_neg_scores = paddle.stack([F.cosine_similarity(query_embeddings, neg_emb, axis=-1) / tau for neg_emb in negative_embeddings], axis=1)
    
    
    all_scores = paddle.concat([pos_scores, all_neg_scores], axis=1)
    
    
    logsumexp_scores = paddle.logsumexp(all_scores, axis=1)
    
    
    loss = -pos_scores + logsumexp_scores
    return loss.mean()
