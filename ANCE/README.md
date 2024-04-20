# ANCE

ANCE（Approximate Nearest Neighbor Negative Contrastive Estimation）是一种用于密集文本检索的学习方法。其核心思想是从整个语料库中选择负样本（即不相关的文档），利用异步更新的最近邻索引（ANN）。这种方法能够从索引中检索出对当前密集检索（DR）模型具有挑战性的“最难”负样本。这些负样本因具有较高的训练损失和梯度范数上限，从而有助于模型的训练收敛​​。

在训练过程中，ANCE使用BERT Siamese/Dual Encoder结构，采用点积相似性和负对数似然（NLL）损失函数。这种方法首先使用预训练的BM25模型生成初始训练数据，然后进行模型训练和ANN索引的周期性更新，以维护索引的实时性​​。


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
@article{xiong2020approximate,
  title={Approximate nearest neighbor negative contrastive learning for dense text retrieval},
  author={Xiong, Lee and Xiong, Chenyan and Li, Ye and Tang, Kwok-Fung and Liu, Jialin and Bennett, Paul and Ahmed, Junaid and Overwijk, Arnold},
  journal={arXiv preprint arXiv:2007.00808},
  year={2020}
}
```

## Checkpoint

链接: https://pan.baidu.com/s/16Ah1HdIqv61Oq09-xvaEhg?pwd=kfir 提取码: kfir