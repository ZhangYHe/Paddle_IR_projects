import paddle
from paddle import tensor
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertTokenizer

class ColBERT(nn.Layer):
    def __init__(self, query_pretrained_model_name_or_path='bert-base-uncased',
                 doc_pretrained_model_name_or_path='bert-base-uncased'):
        super(ColBERT, self).__init__()
        
        self.query_encoder = BertModel.from_pretrained(query_pretrained_model_name_or_path)
        self.doc_encoder = BertModel.from_pretrained(doc_pretrained_model_name_or_path)
    
    def forward(self, input_ids_query, attention_mask_query, input_ids_doc, attention_mask_doc):
        
        query_output = self.query_encoder(input_ids_query, attention_mask=attention_mask_query)
        query_repr = query_output[0]  

        
        doc_output = self.doc_encoder(input_ids_doc, attention_mask=attention_mask_doc)
        doc_repr = doc_output[0]  

        return query_repr, doc_repr
    
    def score(self, query_repr, doc_repr):
        """
        
        query_repr:  [batch_size, query_len, hidden_size]
        doc_repr:  [batch_size, doc_len, hidden_size]
        """
        sim_scores = paddle.matmul(query_repr, doc_repr, transpose_y=True)  # [batch_size, query_len, doc_len]
        max_scores = paddle.max(sim_scores, axis=2)  # [batch_size, query_len]
        relevance_scores = paddle.sum(max_scores, axis=1)  # [batch_size]
        return relevance_scores


def margin_ranking_loss(model, pos_scores, neg_scores, margin=0.3):

    # Margin ranking loss
    losses = paddle.nn.functional.relu(margin - pos_scores + neg_scores)
    return paddle.mean(losses)