# -*- coding: utf-8 -*-
# @Time : 2021/3/30 22:56
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : __init__.py
# @Software: PyCharm
import tensorflow as tf
import time
from config import classifier_config


# class Predictor:
#     def __init__(self, data_manager, logger):
#         hidden_dim = classifier_config['hidden_dim']
#         classifier = classifier_config['classifier']
#         self.dataManager = data_manager
#         self.seq_length = data_manager.max_sequence_length
#         num_classes = data_manager.max_label_number
#         self.embedding_dim = data_manager.embedding_dim
#         vocab_size = data_manager.vocab_size
#
#         self.logger = logger
#         self.checkpoints_dir = classifier_config['checkpoints_dir']
#         self.embedding_method = classifier_config['embedding_method']
#
#         logger.info('loading model parameter')
#
#         self.model = TextCNN(self.seq_length, num_classes, self.embedding_dim, vocab_size)
#         # 实例化Checkpoint，设置恢复对象为新建立的模型
#         checkpoint = tf.train.Checkpoint(model=self.model)
#         # 从文件恢复模型参数
#         checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))
#         logger.info('loading model successfully')
#
#     def predict_one(self, sentence):
#         """
#         对输入的句子分类预测
#         :param sentence:
#         :return:
#         """
#         reverse_classes = {class_id: class_name for class_name, class_id in self.dataManager.class_id.items()}
#         start_time = time.time()
#         vector = self.dataManager.prepare_single_sentence(sentence)
#         logits = self.model.call(inputs=vector)
#         prediction = tf.argmax(logits, axis=-1)
#         prediction = prediction.numpy()[0]
#         self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time)*1000))
#         return reverse_classes[prediction]
#
#     def save_model(self):
#         # 保存pb格式的模型到本地
#         tf.saved_model.save(self.model, self.checkpoints_dir,
#                             signatures=self.model.call.get_concrete_function(
#                                 tf.TensorSpec([None, self.seq_length], tf.float32, name='inputs')))
#         self.logger.info('The model has been saved')
