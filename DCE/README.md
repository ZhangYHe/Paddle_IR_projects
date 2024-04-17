# DCE

DCE模型（Dual Cross Encoder）是一种用于密集文档检索的框架。该模型旨在通过深度的查询交互提升文档的表示多样性，从而改进检索质量和效率。DCE模型特别关注于如何通过交叉编码器来捕捉查询和文档间的深层交互信息。不同于传统的Bi-encoder，该模型能够同时处理查询和文档，通过交叉编码器生成的丰富表示来优化检索效果。通过这种结合了深度语义理解和高效检索机制的方法，DCE模型在多个标准信息检索和问答数据集上都显示出了优越的性能，证明了其在实际应用中的潜力和有效性。


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
@article{li2022learning,
  title={Learning diverse document representations with deep query interactions for dense retrieval},
  author={Li, Zehan and Yang, Nan and Wang, Liang and Wei, Furu},
  journal={arXiv preprint arXiv:2208.04232},
  year={2022}
}
```