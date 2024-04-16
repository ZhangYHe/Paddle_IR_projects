import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertTokenizer

import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel

class COIL_full_Model(nn.Layer):
    def __init__(self, pretrained_model_name='bert-base-uncased', model_name="COIL-full", nt=32):
        super(COIL_full_Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.model_name = model_name  
        self.nt = nt  
        
        self.W_tok = nn.Linear(self.bert.config["hidden_size"], self.nt)
       
        if self.model_name == "COIL-full":
            self.layer_norm = nn.LayerNorm(self.nt)
        print(f"[!] Using {self.model_name}")

    def forward(self, input_ids_query, attention_mask_query, input_ids_passage, attention_mask_passage):
        
        query_repr = self.bert(input_ids_query, attention_mask_query)[0]  
        passage_repr = self.bert(input_ids_passage, attention_mask_passage)[0]  

       
        query_repr = self.W_tok(query_repr)
        passage_repr = self.W_tok(passage_repr)
        
        
        if self.model_name == "COIL-full":
            query_repr = self.layer_norm(query_repr)
            passage_repr = self.layer_norm(passage_repr)
        
        
        dot_product = paddle.matmul(query_repr, passage_repr, transpose_y=True)
        
        
        max_similarity = paddle.max(dot_product, axis=-1)
        
        
        s_tok = paddle.sum(max_similarity, axis=-1)
        
        
        query_cls = query_repr[:, 0, :]  
        passage_cls = passage_repr[:, 0, :]  
        cls_dot_product = paddle.matmul(query_cls, passage_cls, transpose_y=True).squeeze()
        s_full = s_tok + cls_dot_product
        return s_full


def coil_loss(s_pos, s_negs):
    s_pos = s_pos[:, 0]
    
    s_pos = s_pos.unsqueeze(-1)
    s_all = paddle.concat([s_pos, s_negs], axis=1)
    log_softmax_scores = paddle.nn.functional.log_softmax(s_all, axis=1)
    
    pos_log_softmax_scores = log_softmax_scores[:, 0]
    loss = -pos_log_softmax_scores
    
    return paddle.mean(loss)
