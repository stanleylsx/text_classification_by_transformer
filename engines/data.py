# -*- coding: utf-8 -*-
# @Time : 2021/3/30 22:56
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : __init__.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from engines.utils.clean_data import filter_word
from config import classifier_config
from collections import Counter


class DataManager:

    def __init__(self, logger):
        self.logger = logger
        self.embedding_method = classifier_config['embedding_method']
        self.w2v_util = Word2VecUtils(logger)
        self.stop_words = self.w2v_util.get_stop_words()

        self.embedding_dim = classifier_config['embedding_dim']
        self.token_file = classifier_config['token_file']
        if not os.path.isfile(self.token_file):
            self.logger.info('vocab files not exist...')
        else:
            self.word_token2id, self.id2word_token = self.load_vocab()
            self.vocab_size = len(self.word_token2id)

        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'
        self.batch_size = classifier_config['batch_size']
        self.max_sequence_length = classifier_config['max_sequence_length']

        self.class_id = classifier_config['classes']
        self.class_list = [name for name, index in classifier_config['classes'].items()]
        self.max_label_number = len(self.class_id)

        self.logger.info('dataManager initialed...')

    def load_vocab(self, sentences=None):
        if not os.path.isfile(self.token_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab(self.token_file, sentences)
        word_token2id, id2word_token = {}, {}
        with open(self.token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                word_token, word_token_id = row.split('\t')[0], int(row.split('\t')[1])
                word_token2id[word_token] = word_token_id
                id2word_token[word_token_id] = word_token
        self.vocab_size = len(word_token2id)
        return word_token2id, id2word_token

    def build_vocab(self, token_file, sentences):
        word_tokens = []
        for sentence in tqdm(sentences):
            words = self.w2v_util.processing_sentence(sentence, self.stop_words)
            word_tokens.extend(words)
        # 根据词频过滤一部分频率极低的词，不加入词表
        count_dict = Counter(word_tokens)
        word_tokens = [k for k, v in count_dict.items() if v > 1 and filter_word(k)]
        word_token2id = dict(zip(word_tokens, range(1, len(word_tokens) + 1)))
        id2word_token = dict(zip(range(1, len(word_tokens) + 1), word_tokens))
        # 向生成的词表和标签表中加入[PAD]
        id2word_token[0] = self.PADDING
        word_token2id[self.PADDING] = 0
        # 向生成的词表中加入[UNK]
        id2word_token[len(id2word_token)] = self.UNKNOWN
        word_token2id[self.UNKNOWN] = len(id2word_token)
        # 保存词表及标签表
        with open(token_file, 'w', encoding='utf-8') as outfile:
            for idx in id2word_token:
                outfile.write(id2word_token[idx] + '\t' + str(idx) + '\n')
        self.vocab_size = len(word_token2id)
        return word_token2id, id2word_token

    def padding(self, sentence):
        """
        长度不足max_sequence_length则补齐
        :param sentence:
        :return:
        """
        if len(sentence) < self.max_sequence_length:
            sentence += [self.PADDING for _ in range(self.max_sequence_length - len(sentence))]
        else:
            sentence = sentence[:self.max_sequence_length]
        return sentence

    def prepare_data(self, sentences, labels):
        """
        输出X矩阵和y向量
        """
        self.logger.info('loading data...')
        X, y = [], []
        for record in tqdm(zip(sentences, labels)):
            sentence = self.w2v_util.processing_sentence(record[0], self.stop_words)
            sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            word_tokens = []
            for word in sentence:
                if word in self.word_token2id:
                    word_tokens.append(self.word_token2id[word])
                else:
                    word_tokens.append(self.word_token2id[self.UNKNOWN])
            X.append(word_tokens)
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def get_dataset(self, df, step=None):
        """
        构建Dataset
        """
        df = df.loc[df.label.isin(self.class_list)]
        df['label'] = df.label.map(lambda x: self.class_id[x])
        # convert the data in matrix
        if step == 'train' and not os.path.isfile(self.token_file):
            self.word_token2id, self.id2word_token = self.load_vocab(df['sentence'])
        X, y = self.prepare_data(df['sentence'], df['label'])
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
        sentence = self.padding(sentence)
        word_tokens = []
        for word in sentence:
            if word in self.word_token2id:
                word_tokens.append(self.word_token2id[word])
            else:
                word_tokens.append(self.word_token2id[self.UNKNOWN])
        return np.array([word_tokens], dtype=np.float32)
