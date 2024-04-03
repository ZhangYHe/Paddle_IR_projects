import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertTokenizer

class DPRQuestionEncoder(nn.Layer):
    def __init__(self, pretrained_model_name_or_path='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]  # Assume pooler_output is the second element
        return pooler_output

class DPRContextEncoder(nn.Layer):
    def __init__(self, pretrained_model_name_or_path='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]  # Assume pooler_output is the second element
        return pooler_output

class DPR(nn.Layer):
    def __init__(self, question_pretrained_model_name_or_path='bert-base-uncased', context_pretrained_model_name_or_path='bert-base-uncased'):
        super().__init__()
        self.question_encoder = DPRQuestionEncoder(question_pretrained_model_name_or_path)
        self.context_encoder = DPRContextEncoder(context_pretrained_model_name_or_path)

    def forward(self, question_inputs, context_inputs):
        question_embeddings = self.question_encoder(**question_inputs)
        context_embeddings = self.context_encoder(**context_inputs)
        return question_embeddings, context_embeddings

def compute_similarity(question_embeddings, context_embeddings):
    return paddle.matmul(question_embeddings, context_embeddings, transpose_y=True)
