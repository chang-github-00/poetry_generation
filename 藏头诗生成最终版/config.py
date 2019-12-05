'''
configurations
'''

import os

class Config(object):
    poetry_file = os.path.join('.', 'data', 'poetry.txt')
    weight_file = os.path.join('.', 'model', 'poetry_model.h5')
    preprocess_file = os.path.join('.', 'data', 'data')
    ryhme_file = os.path.join('.', 'data', '13zhe.txt')
    embedding_file = os.path.join('.', 'data', 'sgns.sikuquanshu.word')
    tb_file = os.path.join('.', 'tensorboard')
    # 根据前六个字预测第七个字
    max_len = 6
    # 字典中每个字的最少出现次数
    word_min_num = 2
    batch_size = 5000
    learning_rate = 0.001
    epoch_num = 240

    embedding_size = 300
    rnn_1_size = 128
    rnn_2_size = 256

    dropout_rate = 0.6
