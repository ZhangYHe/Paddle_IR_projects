# COIL-tok

COIL-tok (Contextualized Inverted List with token-only model) 是一种信息检索模型，主要侧重于利用BERT等预训练语言模型对查询和文档中的单个词汇进行编码，然后计算这些词汇之间的相似性得分。COIL-tok不涉及文档级的语义信息，它通过计算查询和文档中每个词汇的点积相似度，并通过最大池化操作提取最有价值的特征，以此来评估查询和文档之间的相关性。这种模型结构简单，主要用于那些侧重词汇级匹配的场景，能够有效捕捉查询和文档之间的细粒度相似性。

# Environment
- python 3.10
- paddle 2.6.1
- cuda 12.0
- cudnn 8.9.1

## Dataset 
- [MS MARCO - Passage ranking dataset](https://microsoft.github.io/msmarco/Datasets#passage-ranking-dataset)

# Train

```
  bash train.sh
```

# Inference
```
  bash test.sh
```

## Reference 
```
@article{gao2021coil,
  title={COIL: Revisit exact lexical match in information retrieval with contextualized inverted list},
  author={Gao, Luyu and Dai, Zhuyun and Callan, Jamie},
  journal={arXiv preprint arXiv:2104.07186},
  year={2021}
}
```

## Checkpoint

链接: https://pan.baidu.com/s/16Ah1HdIqv61Oq09-xvaEhg?pwd=kfir 提取码: kfir