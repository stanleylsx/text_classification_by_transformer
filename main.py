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
from config import mode, classifier_config
import json


if __name__ == '__main__':
    logger = get_logger('./logs')
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
        predictor.predict_one('2010年6月英语四级深度阅读答案(昂立)以下是2010年6月19日四级答案 深度阅读十五选十：47. incredibly48. replace49. reduced50. sense51. powering52. exceptions53. expand54. vast55. historic56. protect来源：昂立教育 王如卿')
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
