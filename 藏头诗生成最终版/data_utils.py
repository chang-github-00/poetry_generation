'''
预处理文本内容
'''
import os
import pickle
import numpy as np


def preprocess_file(config):
    '''
    :param config:
    :return (word2numF, num2word, words, file_content):
    预处理
    '''
    data = ()
    if os.path.exists(config.preprocess_file):
        with open(config.preprocess_file, 'rb') as f:
            data = pickle.load(f)
    else:
        # 要去掉的字符
        puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']
        # 语料文本内容
        files_content = ''
        with open(config.poetry_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 每行的末尾加上"]"符号代表一首诗结束
                for char in puncs:
                    line = line.replace(char, "")
                files_content += line.strip() + "]"

        words = sorted(list(files_content))
        words.remove(']')
        counted_words = {}
        for word in words:
            if word in counted_words:
                counted_words[word] += 1
            else:
                counted_words[word] = 1

        # 去掉低频的字
        erase = []
        for key in counted_words:
            if counted_words[key] <= config.word_min_num:
                erase.append(key)
        for key in erase:
            del counted_words[key]
        del counted_words[']']
        wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

        words, _ = zip(*wordPairs)
        # word到id的映射
        word2num = dict((c, i + 1) for i, c in enumerate(words))
        num2word = dict((i, c) for i, c in enumerate(words))
        data = (word2num, num2word, words, files_content)
        with open(config.preprocess_file, 'wb') as f:
           pickle.dump(data, f)

    word2numF = lambda x: data[0].get(x, 0)
    return word2numF, data[1], data[2], data[3]


def embedding_file(config, words, embedding_size=300):
    embeddings_index = {}
    with open(config.embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(words) + 2, embedding_size))
    for i, word in enumerate(words):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
