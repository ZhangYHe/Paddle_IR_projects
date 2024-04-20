# Contriever

Contriever模型是一种先进的、无监督的密集信息检索模型，由Gautier Izacard等人提出，其详细描述见于论文《Unsupervised Dense Information Retrieval with Contrastive Learning》。Contriever模型使用对比学习方法，在没有任何监督信息的情况下训练密集检索器，并在各种检索设置中展现出了强大的性能。该模型特别强调了对比学习在无监督训练密集检索器中的潜力，并通过在BEIR基准测试中的表现证明了其有效性。相对于传统的基于术语频率的方法（如BM25），Contriever在11个中的15个数据集上的Recall@100指标表现更优。

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
@article{izacard2021unsupervised,
  title={Unsupervised dense information retrieval with contrastive learning},
  author={Izacard, Gautier and Caron, Mathilde and Hosseini, Lucas and Riedel, Sebastian and Bojanowski, Piotr and Joulin, Armand and Grave, Edouard},
  journal={arXiv preprint arXiv:2112.09118},
  year={2021}
}
```

## Checkpoint

链接: https://pan.baidu.com/s/16Ah1HdIqv61Oq09-xvaEhg?pwd=kfir 提取码: kfir