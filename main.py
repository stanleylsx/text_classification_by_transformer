# -*- coding: utf-8 -*-
# @Time : 2021/3/30 22:56
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : __init__.py
# @Software: PyCharm
from engines.data import DataManager
from engines.utils.logger import get_logger
from engines.train import train
from engines.predict import Predictor
from config import mode, classifier_config, CUDA_VISIBLE_DEVICES
import json
import os


if __name__ == '__main__':
    logger = get_logger('./logs')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
    # 训练分类器
    if mode == 'train_classifier':
        logger.info(json.dumps(classifier_config, indent=2))
        data_manage = DataManager(logger)
        logger.info('mode: train_classifier')
        train(data_manage, logger)
    # 测试分类
    elif mode == 'interactive_predict':
        logger.info(json.dumps(classifier_config, indent=2))
        data_manage = DataManager(logger)
        logger.info('mode: predict_one')
        predictor = Predictor(data_manage, logger)
        predictor.predict_one('warm start')
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            results = predictor.predict_one(sentence)
            print(results)
    # 保存pb格式的模型用于tf-severing接口
    elif mode == 'save_model':
        logger.info('mode: save_pb_model')
        data_manage = DataManager(logger)
        predictor = Predictor(data_manage, logger)
        predictor.save_model()
