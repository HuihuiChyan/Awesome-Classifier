# Awesome-Classifier
基于BERT的文本分类器，原本用于评论分类任务
* 输入字段包括多个特征，进行了简单的拼接
* 由于输入长度超过512，因此尝试了不同的截断策略
* 分类类别是多个二分类任务，也是仅仅使用了不同位置的输出

## 模型架构如下：

<img src="image.png" width = "500" height = "350" alt="图片名称" align=center />
