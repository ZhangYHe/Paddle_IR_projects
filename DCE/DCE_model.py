import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertTokenizer

class DCE_model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.query_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.document_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, query_text, document_text):
        
        query_inputs = self.tokenizer(
            query_text, 
            max_length=128, 
            padding='max_length',  
            truncation=True,       
            return_tensors='pd'    
        )

        query_outputs = self.query_encoder(**query_inputs)
        query_cls_embedding = query_outputs[1]  

        
        combined_text = [q + " " + d for q, d in zip(query_text, document_text)]
        document_inputs = self.tokenizer(
            combined_text, 
            max_length=512, 
            padding='max_length',  
            truncation=True,       
            return_tensors='pd'    
        )
        document_outputs = self.document_encoder(**document_inputs)
        document_cls_embedding = document_outputs[1]  

        
        similarity_score = paddle.sum(query_cls_embedding * document_cls_embedding, axis=1)

        return similarity_score

def DCE_loss(pos_score, neg_scores):
    """
    param pos_score: [batch_size]。
    param neg_scores: [batch_size, num_neg_samples]
    """
    
    scores = paddle.concat([pos_score.unsqueeze(1), neg_scores], axis=1)
    
    
    log_softmax_scores = paddle.nn.functional.log_softmax(scores, axis=1)
    
    
    log_pos_probs = log_softmax_scores[:, 0]
    
    
    loss = -log_pos_probs
    
    
    return paddle.mean(loss)
