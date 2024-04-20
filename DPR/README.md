# DPR (Dense Passage Retrieval)

DPR是一种用于信息检索的模型，它通过将问题和文档嵌入到同一向量空间中，然后通过计算它们之间的向量相似性来检索相关文档。

DPR模型由两部分组成：**问题编码器**和**文档编码器**。问题编码器将用户的问题转化为向量，文档编码器将文档转化为向量。然后，通过计算问题向量和文档向量之间的点积，可以得到问题和文档之间的相似性得分。

## Environment
- python 3.10
- paddle 2.6.1
- cuda 12.0
- cudnn 8.9.1

## Dataset 
- [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)

## Train

```
  bash train.sh
```

## Inference
```
  bash test.sh
```

## Reference 
```
@article{karpukhin2020dense,
  title={Dense passage retrieval for open-domain question answering},
  author={Karpukhin, Vladimir and O{\u{g}}uz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau},
  journal={arXiv preprint arXiv:2004.04906},
  year={2020}
}
```

## Checkpoint

链接: https://pan.baidu.com/s/16Ah1HdIqv61Oq09-xvaEhg?pwd=kfir 提取码: kfir