# Text Classification By Transformer

此仓库是使用transformer模型进行文本分类的任务，项目基于Tensorflow2.3开发，配置参数之后，开箱即用。

## 更新历史
日期|版本|描述
:---|:---|---
2021-03-30|v1.0.0|初始仓库
2021-04-15|v1.1.0|解决模型不收敛的bug

## 主要环境
* python 3.7.10
* **CPU:** tensorflow==2.3.0
* **GPU:** tensorflow-gpu==2.3.0

## 数据集
网络新闻数据集

## 原理
使用Transformer的Encoder做文本的分类。下图左部分结构:  

![Transformer](https://img-blog.csdnimg.cn/20210416114817619.jpg)

Encoder结构:

![encoder](https://img-blog.csdnimg.cn/20210416114817385.png)

## 配置
配置好下列参数    
```
classifier_config = {
    # 训练数据集
    'train_file': 'data/news_data/train_data.csv',
    # 验证数据集
    'dev_file': 'data/news_data/val_data.csv',
    # 向量维度
    'embedding_dim': 240,
    # 存放字/词表的地方
    'token_file': 'data/news_data/char_token2id',
    # 类别和对应的id
    'classes': {'体育': 0, '房产': 1, '财经': 2, '科技': 3, '时政': 4, '时尚': 5, '游戏': 6, '教育': 7, '娱乐': 8, '家居': 9},
    # 停用词(可为空)
    'stop_words': 'data/news_data/stop_words.txt',
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoints/news_classification',
    # 模型保存的名字
    'checkpoint_name': 'tf-model',
    # token粒度
    # 词粒度:'word'
    # 字粒度:'char'
    'token_level': 'char',
    # 学习率
    'learning_rate': 1e-3,
    # 训练epoch
    'epoch': 30,
    # 最多保存max_to_keep个模型
    'max_to_keep': 1,
    # 每print_per_batch打印
    'print_per_batch': 20,
    # 是否提前结束
    'is_early_stop': True,
    'patient': 8,
    'batch_size': 64,
    'max_sequence_length': 100,
    # Encoder的个数
    'encoder_num': 1,
    # 遗忘率
    'dropout_rate': 0.25,
    # 多头注意力的个数
    'head_num': 12,
    # 隐藏层维度
    'hidden_dim': 2048,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'micro',
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': False
}
```

## 训练与预测
### 训练
```
# [train_classifier, interactive_predict, train_word2vec]
mode = 'train_classifier'
```
![train](https://img-blog.csdnimg.cn/20210416144659936.png)

### 交互测试
```
mode = 'interactive_predict'
```
![test](https://img-blog.csdnimg.cn/20210416144713956.png)