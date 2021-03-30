# -*- coding: utf-8 -*-
# @Time : 2021/3/30 22:56
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : __init__.py
# @Software: PyCharm


# [train_classifier, interactive_predict, save_model]
mode = 'interactive_predict'

classifier_config = {
    # 模型选择
    'classifier': 'textcnn',
    # 训练数据集
    'train_file': 'data/data/train_data.csv',
    # 引入外部的词嵌入,可选word2vec、Bert
    # 此处只使用Bert Embedding,不对其做预训练
    # None:使用随机初始化的Embedding
    'embedding_method': 'word2vec',
    # 不外接词向量的时候需要自定义的向量维度
    'embedding_dim': 300,
    # 存放词表的地方
    'token_file': 'data/data/token2id',
    # 验证数据集
    'dev_file': 'data/data/dev_data.csv',
    # 类别和对应的id
    'classes': {'negative': 0, 'positive': 1},
    # 模型保存的文件夹
    'checkpoints_dir': 'model/word2vec_textcnn',
    # 模型保存的名字
    'checkpoint_name': 'word2vec_textcnn',
    # 卷集核的个数
    'num_filters': 64,
    # 学习率
    'learning_rate': 0.001,
    # 训练epoch
    'epoch': 30,
    # 最多保存max_to_keep个模型
    'max_to_keep': 1,
    # 每print_per_batch打印
    'print_per_batch': 20,
    # 是否提前结束
    'is_early_stop': True,
    # 是否引入attention
    # 注意:textrcnn不支持
    'use_attention': False,
    # attention大小
    'attention_size': 300,
    'patient': 8,
    'batch_size': 64,
    'max_sequence_length': 150,
    # 遗忘率
    'dropout_rate': 0.5,
    # 隐藏层维度
    # 使用textrcnn中需要设定
    'hidden_dim': 200,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'binary',
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': False
}
