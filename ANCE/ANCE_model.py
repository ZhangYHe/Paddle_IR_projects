import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertTokenizer

class ANCE(paddle.nn.Layer):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(ANCE, self).__init__()
        self.query_encoder = BertModel.from_pretrained(pretrained_model_name)
        self.doc_encoder = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids_query, input_ids_doc, attention_mask_query=None, attention_mask_doc=None):
        if input_ids_query is None:
            query_emb = None
        else:
            query_emb = self.query_encoder(input_ids_query, attention_mask=attention_mask_query)[1]
        doc_emb = self.doc_encoder(input_ids_doc, attention_mask=attention_mask_doc)[1]
        return query_emb, doc_emb


def NLL_loss(query_emb, pos_emb, neg_emb):
    pos_score = paddle.sum(query_emb * pos_emb, axis=1)
    neg_score = paddle.sum(query_emb * neg_emb, axis=1)

    # log-sum-exp
    max_val = paddle.maximum(pos_score, neg_score)
    log_sum_exp = max_val + paddle.log(paddle.exp(pos_score - max_val) + paddle.exp(neg_score - max_val))
    loss = log_sum_exp - pos_score
    return paddle.mean(loss)
