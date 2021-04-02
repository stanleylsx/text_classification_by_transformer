# -*- coding: utf-8 -*-
# @Time : 2021/3/30 22:56
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : __init__.py
# @Software: PyCharm


# [train_classifier, interactive_predict, save_model]
mode = 'train_classifier'

classifier_config = {
    # 训练数据集
    'train_file': 'data/train_data.csv',
    # 验证数据集
    'dev_file': 'data/dev_data.csv',
    # 向量维度
    'embedding_dim': 300,
    # 存放词表的地方
    'token_file': 'data/token2id',
    # 类别和对应的id
    'classes': {'negative': 0, 'positive': 1},
    # 停用词(可为空)
    'stop_words': 'data/w2v_data/stop_words.txt',
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoints/model',
    # 模型保存的名字
    'checkpoint_name': 'model',
    # token粒度
    'token_level': 'word',
    # 学习率
    'learning_rate': 0.005,
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
    'max_sequence_length': 20,
    # Encoder的个数
    'encoder_num': 2,
    # 遗忘率
    'dropout_rate': 0.5,
    # 多头注意力的个数
    'head_num': 5,
    # 隐藏层维度
    'hidden_dim': 1024,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'binary',
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': False
}
