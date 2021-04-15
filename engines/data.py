# -*- coding: utf-8 -*-
# @Time : 2021/3/30 22:56
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : __init__.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import os
import jieba
from tqdm import tqdm
from engines.utils.clean_data import filter_word, filter_char
from config import classifier_config
from collections import Counter


class DataManager:

    def __init__(self, logger):
        self.logger = logger
        self.token_level = classifier_config['token_level']
        self.stop_words = self.get_stop_words()

        self.embedding_dim = classifier_config['embedding_dim']
        self.token_file = classifier_config['token_file']
        if not os.path.isfile(self.token_file):
            self.logger.info('vocab files not exist...')
        else:
            self.token2id, self.id2token = self.load_vocab()
            self.vocab_size = len(self.token2id)

        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'
        self.batch_size = classifier_config['batch_size']
        self.max_sequence_length = classifier_config['max_sequence_length']

        self.class_id = classifier_config['classes']
        self.class_list = [name for name, index in classifier_config['classes'].items()]
        self.max_label_number = len(self.class_id)

        self.logger.info('dataManager initialed...')

    @staticmethod
    def processing_sentence(x, stop_words):
        cut_word = jieba.cut(str(x).strip())
        if stop_words:
            words = [word for word in cut_word if word not in stop_words and word != ' ']
        else:
            words = list(cut_word)
            words = [word for word in words if word != ' ']
        return words

    @staticmethod
    def get_stop_words():
        stop_words_path = classifier_config['stop_words']
        stop_words_list = []
        try:
            with open(stop_words_path, 'r', encoding='utf-8') as stop_words_file:
                for line in stop_words_file:
                    stop_words_list.append(line.strip())
        except FileNotFoundError:
            return stop_words_list
        return stop_words_list

    def load_vocab(self, sentences=None):
        if not os.path.isfile(self.token_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab(self.token_file, sentences)
        token2id, id2token = {}, {}
        with open(self.token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
        self.vocab_size = len(token2id)
        return token2id, id2token

    def build_vocab(self, token_file, sentences):
        tokens = []
        if self.token_level == 'word':
            # 词粒度
            for sentence in tqdm(sentences):
                words = self.processing_sentence(sentence, self.stop_words)
                tokens.extend(words)
            # 根据词频过滤一部分频率极低的词/字，不加入词表
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if v > 1 and filter_word(k)]
        else:
            # 字粒度
            for sentence in tqdm(sentences):
                chars = list(sentence)
                tokens.extend(chars)
            # 根据词频过滤一部分频率极低的词/字，不加入词表
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if k != ' ' and filter_char(k)]
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        # 向生成的词表和标签表中加入[PAD]
        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0
        # 向生成的词表中加入[UNK]
        id2token[len(id2token)] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(id2token)
        # 保存词表及标签表
        with open(token_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')
        self.vocab_size = len(token2id)
        return token2id, id2token

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
            if self.token_level == 'word':
                sentence = self.processing_sentence(record[0], self.stop_words)
                sentence = self.padding(sentence)
            else:
                sentence = list(record[0])
                sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            tokens = []
            for token in sentence:
                if token in self.token2id:
                    tokens.append(self.token2id[token])
                else:
                    tokens.append(self.token2id[self.UNKNOWN])
            X.append(tokens)
            y.append(label)
        return np.array(X), np.array(y)

    def get_dataset(self, df, step=None):
        """
        构建Dataset
        """
        df = df.loc[df.label.isin(self.class_list)]
        df['label'] = df.label.map(lambda x: self.class_id[x])
        # convert the data in matrix
        if step == 'train' and not os.path.isfile(self.token_file):
            self.token2id, self.id2token = self.load_vocab(df['sentence'])
        X, y = self.prepare_data(df['sentence'], df['label'])
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        tokens = []
        if self.token_level == 'word':
            sentence = self.processing_sentence(sentence, self.stop_words)
            sentence = self.padding(sentence)
        else:
            sentence = list(sentence)
            sentence = self.padding(sentence)
        for token in sentence:
            if token in self.token2id:
                tokens.append(self.token2id[token])
            else:
                tokens.append(self.token2id[self.UNKNOWN])
        return np.array([tokens])
