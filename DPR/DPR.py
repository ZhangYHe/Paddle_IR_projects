import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertTokenizer

class DPRQuestionEncoder(nn.Layer):
    def __init__(self, pretrained_model_name_or_path='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # 修改这里，通过索引访问 pooler_output
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]  # 通过索引访问，假设pooler_output是第二个元素
        return pooler_output

class DPRContextEncoder(nn.Layer):
    def __init__(self, pretrained_model_name_or_path='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # 修改这里，通过索引访问 pooler_output
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]  # 通过索引访问，假设pooler_output是第二个元素
        return pooler_output


def compute_similarity(question_embeddings, context_embeddings):
    return paddle.matmul(question_embeddings, context_embeddings, transpose_y=True)

# 示例数据加载和处理
# 这里简单地使用假数据来演示如何训练模型
questions = ["What is PaddlePaddle?"]
contexts = ["PaddlePaddle is an open-source deep learning platform."]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

question_inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pd")
context_inputs = tokenizer(contexts, padding=True, truncation=True, return_tensors="pd")

# 定义模型和优化器
question_encoder = DPRQuestionEncoder()
context_encoder = DPRContextEncoder()
optimizer = paddle.optimizer.Adam(parameters=question_encoder.parameters() + context_encoder.parameters(), learning_rate=5e-5)

# 模拟训练循环
for epoch in range(10):
    question_encoder.train()
    context_encoder.train()

    question_embeddings = question_encoder(**question_inputs)
    context_embeddings = context_encoder(**context_inputs)

    # 假设正样本位于对角线上，计算相似度得分
    sim_scores = compute_similarity(question_embeddings, context_embeddings)
    labels = paddle.arange(sim_scores.shape[0])
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(sim_scores, labels)

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    print(f"Epoch {epoch}, Loss: {loss.numpy().item()}")
