# ColBERT

ColBERT是一种基于BERT的高效信息检索模型，旨在解决大规模检索任务中的准确性和效率问题。通过引入一种名为“Late Interaction”的技术，ColBERT能够在保持深度语义理解能力的同时，显著提高检索速度。ColBERT独特地对查询和文档的每个词生成独立的向量表示，而不是将整个查询或文档映射到单一的密集向量。这种细粒度表示允许模型在词级别上进行更精细的相似度计算。

在绝大多数基于BERT的检索模型中，查询和文档是先合并再输入到模型中的。与之不同，ColBERT采用Late Interaction机制，在生成词向量表示之后再进行交互计算，这样可以大幅降低计算成本。通过以上机制，ColBERT在不牺牲准确性的前提下，显著提高了检索效率，使其能够应对大规模信息检索任务。


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
@inproceedings{khattab2020colbert,
  title={Colbert: Efficient and effective passage search via contextualized late interaction over bert},
  author={Khattab, Omar and Zaharia, Matei},
  booktitle={Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval},
  pages={39--48},
  year={2020}
}
```

## Checkpoint

链接: https://pan.baidu.com/s/16Ah1HdIqv61Oq09-xvaEhg?pwd=kfir 提取码: kfir