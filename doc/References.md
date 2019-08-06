## References

[Google Cloud AutoML Natural Language Example](https://www.youtube.com/watch?v=-zteIdpQ5UE)

[Make PySpark Work with spaCy: Overcoming Serialization Errors](https://blog.dominodatalab.com/making-pyspark-work-spacy-overcoming-serialization-errors/)

[Topic Modelling with spaCy and scikit-learn](https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn)

[Document Similarity, Tokenization and Word Vectors in Python with spaCy](https://ai.intelligentonlinetools.com/ml/document-similarity/)

[Language Processing Pipelines](https://spacy.io/usage/processing-pipelines)

[NLP tasks](https://en.wikipedia.org/wiki/Natural_language_processing)

[Topic Model Evaluation in Python with Tmtoolkit](https://datascience.blog.wzb.eu/2017/11/09/topic-modeling-evaluation-in-python-with-tmtoolkit/)

[Topic Coherence](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/)

[Evaluation Methods for Topic Models](https://www.cs.cmu.edu/~rsalakhu/papers/etm.pdf)

[LDA on Apache Spark](https://databricks.com/blog/2015/09/22/large-scale-topic-modeling-improvements-to-lda-on-apache-spark.html)

[David Blei](https://scholar.google.de/citations?hl=en&user=8OYE6iEAAAAJ&view_op=list_works)

[Spacy with Gensim LDA](https://medium.com/@colemiller94/topic-modeling-with-spacy-and-gensim-7ecfd3de95f4)

[Topic Evaluation](https://www.quora.com/What-are-good-ways-of-evaluating-the-topics-generated-by-running-LDA-on-a-corpus)

Keyword: LDA; Spark; NMF, LSA (LSI); pLSA (pLSI) hLSA; HDP ; GMM; topic modeling evaluation

pLSA; BTM; TMM 被认为是LDA的低配版。

理解LDA：

拜叶斯统计（先验概率）

理解pLSA:

最大似然估计；EM算法

pLSA的建模思想较为简单，对于observed variables建立likelihood function，将latent variable暴露出来，并使用E-M算法求解。其中M步的标准做法是引入Lagrange乘子求解后回代到E步。

但pLSA将![[公式]](https://www.zhihu.com/equation?tex=p%28z_k%7Cd_m%29)与![[公式]](https://www.zhihu.com/equation?tex=p%28w_n%7Cz_k%29)看做未知常量，存在难以incorporate先验知识的问题，故被贝叶斯学派拿来进行了改造。

pLSA中，文档-主题分布以及主题-单词分布都是确定的，不是随机变量，而在LDA中，这两个分布都是各自有一个狄利克雷的先验分布的。

* **短文本**就词共现来说存在极大的稀疏性。数据稀疏性成为了提高短文本主题模型结果的瓶颈。