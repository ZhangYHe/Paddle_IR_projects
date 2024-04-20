# COIL-full

COIL-full (Contextualized Inverted List with full context model) 计算词汇级的相似性，还加入了文档级的语义匹配。在COIL-full模型中，除了使用预训练语言模型对查询和文档中的词汇进行编码和计算相似度外，还特别引入了对CLS向量的处理。CLS向量通常被用作句子或文档的整体语义表示，在COIL-full中，通过计算查询和文档的CLS向量之间的点积来评估它们的整体语义相关性。此外，COIL-full还包括了层归一化（Layer Normalization）操作，以提高模型的训练稳定性和性能。这使得COIL-full在处理包含丰富语义信息的复杂查询时表现更为出色，能够更全面地理解查询意图和文档内容。

## Environment
- python 3.10
- paddle 2.6.1
- cuda 12.0
- cudnn 8.9.1

## Dataset 
- [MS MARCO - Passage ranking dataset](https://microsoft.github.io/msmarco/Datasets#passage-ranking-dataset)

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
@article{gao2021coil,
  title={COIL: Revisit exact lexical match in information retrieval with contextualized inverted list},
  author={Gao, Luyu and Dai, Zhuyun and Callan, Jamie},
  journal={arXiv preprint arXiv:2104.07186},
  year={2021}
}
```

## Checkpoint

链接: https://pan.baidu.com/s/16Ah1HdIqv61Oq09-xvaEhg?pwd=kfir 提取码: kfir