import random
import os

import keras
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, GRU
from keras.optimizers import Adam
import math

from data_utils import *


class PoetryModel(object):
    def __init__(self, config, gr):
        self.model = None
        self.config = config
        self.gr = gr

        # 文件预处理
        self.word2numF, self.num2word, self.words, self.files_content = preprocess_file(self.config)

        # 加载embedding
        self.embedding_matrix = embedding_file(self.config, self.words)

        # 加载模型
        if os.path.exists(self.config.weight_file):
            self.model = load_model(self.config.weight_file)
        else:
            self.build_model()
        self.model.summary()

    def build_model(self):
        '''
        建立模型
        '''
        print("---building new model---")
        cfg = self.config
        # 输入的dimension
        input_tensor = Input(shape=(cfg.max_len,))
        embedd = Embedding(len(self.num2word) + 2,
                           cfg.embedding_size,
                           weights=[self.embedding_matrix],
                           input_length=cfg.max_len,
                           trainable=False)(input_tensor)
        lstm = Bidirectional(GRU(cfg.rnn_1_size, return_sequences=True))(embedd)
        dropout = Dropout(cfg.dropout_rate)(lstm)
        lstm = GRU(cfg.rnn_2_size, return_sequences=True)(dropout)
        dropout = Dropout(cfg.dropout_rate)(lstm)
        flatten = Flatten()(dropout)
        dense = Dense(len(self.words)+1, activation='softmax')(flatten)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=cfg.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def data_generator(self, batch_size):
        '''
        数据生成器
        '''
        data = self.files_content
        i, data_len, max_len = 0, len(data), self.config.max_len
        while True:
            x_data, y_data = [], []
            for _ in range(batch_size):
                while i + max_len < data_len and data[i: min(data_len - 1, i + max_len + 1)].find(']') != -1: i += 1
                if i + max_len == data_len: break

                x = self.files_content[i: i + self.config.max_len]
                y = self.files_content[i + self.config.max_len]

                y_vec = np.zeros(
                    shape=(len(self.words)+1,),
                    dtype=np.bool
                )
                y_vec[self.word2numF(y)] = 1.0

                x_vec = np.zeros(
                    shape=(max_len,),
                    dtype=np.int32
                )
                for t, char in enumerate(x):
                    x_vec[t] = self.word2numF(char)

                x_data.append(x_vec)
                y_data.append(y_vec)
                i += 1
            if len(x_data) == 0:
                i = 0
                continue
            yield np.array(x_data), np.array(y_data)
        return

    def train(self, epoch_num):
        '''
        训练模型
        '''
        def calc_number_of_batch(data, batch_size, max_len):
            cnt = 0
            data_len = len(data)
            for i in range(data_len):
                if i + max_len < data_len and data[i: min(data_len-1, i + max_len + 1)].find(']') == -1: cnt += 1
            print(len(data), cnt, int(math.ceil(cnt / batch_size)))
            return int(math.ceil(cnt / batch_size))
        number_of_batch = calc_number_of_batch(self.files_content, self.config.batch_size, self.config.max_len)

        tbCallback = keras.callbacks.TensorBoard(log_dir=self.config.tb_file,
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)
        self.model.fit_generator(
            generator=self.data_generator(self.config.batch_size),
            verbose=True,
            steps_per_epoch=number_of_batch,
            epochs=epoch_num,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result),
                tbCallback
            ]
        )

    def generate_sample_result(self, epoch, logs):
        '''
        训练过程中，每个epoch打印出当前的学习情况
        '''
        # if epoch % 5 != 0:
        #     return
        print("\n==================Epoch {}=====================".format(epoch+1))
        for diversity in [0.5, 1.0, 1.5]:
            print("------------Diversity {}--------------".format(diversity))
            start_index = random.randint(0, len(self.files_content) - self.config.max_len - 1)
            generated = ''
            sentence = self.files_content[start_index: start_index + self.config.max_len]
            generated += sentence
            for i in range(20):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(sentence[-self.config.max_len:]):
                    x_pred[0, t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.num2word[next_index]

                generated += next_char
                sentence = sentence + next_char
            print(sentence)

    def sample(self, preds, temperature=1.0, nline=0, pos=2, p_model=0, used=list()):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        '''

        '''
        平起首句押韵平平仄仄平，仄仄仄平平。仄仄平平仄，平平仄仄平。
        仄起首句押韵仄仄仄平平，平平仄仄平。平平平仄仄，仄仄仄平平
        '''
        # 改为下标从0开始
        preds = preds[1:]

        pm = [[[0, 0, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]],
            [[1, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 1, 1, 0, 0]]]

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        if pos == 5:
            if nline & 1:
                return self.word2numF('，')
            else:
                return self.word2numF('，')

        ans = -1
        maxp = 0
        pre = preds.tolist()
        for n in range(len(pre)):
            thischar = self.num2word[n]
            # print(thischar)
            if thischar not in self.gr.rhyme.keys():
                continue
            if thischar not in used and pre[n] > maxp and \
                    (nline == 0 or pos != 4 or self.gr.rhyme[thischar] == self.gr.rhy) and \
                    self.gr.pitch[thischar] == pm[p_model][nline][pos]:
                maxp = pre[n]
                ans = n

        if ans == -1:
            # print('not found')
            probas = np.random.multinomial(1, preds, 1)
            ans = np.argmax(probas)
            if nline == 0 and pos == 4:
                while self.num2word[ans] not in self.gr.rhyme.keys():
                    probas = np.random.multinomial(1, preds, 1)
                    ans = np.argmax(probas)
        
        if nline == 0 and pos == 4:
            self.gr.rhy = self.gr.rhyme[self.num2word[ans]]
        
        # print(nline, pos, self.num2word[ans], 'rhy = ',
        #      self.gr.rhyme[self.num2word[ans]], 'rhyme should be ',
        #      self.gr.rhy, 'pitch = ', self.gr.pitch[self.num2word[ans]], 'pitch should be ', pm[p_model][nline][pos])
        
        return ans

    def predict(self, text):
        '''
        根据给出的文字，生成诗句
        '''
        def is_valid(text):
            if len(text) != 4: return False
            if text[0] not in self.gr.rhyme.keys(): return False
            return True
        if not is_valid(text):
            return 'illegal input'
        used = list()
        self.gr.rhy = -1
        with open(self.config.poetry_file, 'r', encoding='utf-8') as f:
            file_list = f.readlines()
        random_line = random.choice(file_list)

        pitch_model = 0
        if self.gr.pitch[text[0]] == 0:
            pass
        else:
            pitch_model = 1

        # 如果给的text不到四个字，则随机补全
        if not text or len(text) != 4:
            for _ in range(4 - len(text)):
                random_str_index = random.randrange(0, len(self.words))
                if self.num2word.get(random_str_index) not in [',', '。', '，']:
                    text += self.num2word.get(random_str_index)
                else:
                    text += self.num2word.get(random_str_index + 1)
        seed = random_line[-self.config.max_len:-1]

        res = ''

        seed = 'c' + seed
        nl = 0
        for c in text:
            seed = seed[1:] + c
            if nl:
                for i in range(0, 5):
                    used.append(seed[i])
            for j in range(5):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(seed):
                    x_pred[0, t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, 1.0, nl, j+1, pitch_model, used)
                # print(next_index)
                next_char = self.num2word[next_index]
                seed = seed[1:] + next_char
            res += seed
            nl += 1
        res = res[:5]+','+res[6:17]+','+res[18:]
        return res


def main():
    from config import Config
    from get_ryhme import Ryhme

    model = PoetryModel(Config, Ryhme(Config))

    # 确定是否训练
    opt = int(input('please input the number of epochs to train or 0 to write a poem\n'))
    if opt != 0:
        model.train(opt)
    model.predict('蛤蛤蛤蛤')

    # 念诗
    while True:
        text = input('text: (输入*退出)')
        if text == '*': break

        sentence = model.predict(text)
        print(sentence)


if __name__ == '__main__':
    main()
