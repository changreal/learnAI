import os
import pickle
import numpy as np
import random

import keras
from keras.models import load_model, Input, Model
from keras.layers import Embedding, LSTM, Dropout, Dense, GRU, Bidirectional, Flatten
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback

# 自定义文件
from data_utils import *

"""
优化：
数据只使用五言绝句
修复bugs
精简代码
训练时测试数据会写入out.txt
训练时的测试输入为随机一首诗的开头，确保输出好看！
添加多个模型方法，可按需要生成诗句

"""

"""
构建模型
"""
class PoetryModel(object):

    def __init__(self, config):
        """
        通过加载Config配置信息，进行语料预处理、模型加载；
        如果训练后的模型文件存在，则直接加载模型，否则开始训练
        :param config:
        """
        self.model = None          # 是否建模了
        # self.do_train = True
        self.loaded_model = False  # 是否加载了模型h5模型
        self.config = config

        # 文件预处理
        self.word2numF , self.num2word, self.words, self.files_content =  preprocess_file(self.config)
        # print(len(self.words))  # 5551

        # ++++添加++++
        self.poems = self.files_content.split(']')  # 诗的list
        self.poems_num = len(self.poems)  # 诗的总数量

        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.config.weight_file):
            self.model = load_model(self.config.weight_file)
            self.model.summary()
        else:
            self.train()

        # self.do_train = False    # ？
        self.loaded_model = True  # 因为之前训练过了，所以现在True了

    # 生成数据
    def data_generator(self):
        i = 0
        # 虽然是whil1，但实际上在生成器中，i+max_len > len(file_content)的时候，下标已经超过语料长度，所以后面要限制模型学习的循环次数
        # 上面数组溢出问题待解决！
        while 1:

            x = self.files_content[i : i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]  # 这里小标要注意哦

            # 排除掉不是同一首诗的
            if ']' in x or ']' in y:   #  对诗结束做处理
                i += 1
                continue

            # 模型学习数据表示：用one-hot表示x、y向量， 每次6个字+1个字的one-hot numpy向量表示
            # x是三维数组、y是二维数组、
            y_vec = np.zeros(
                shape = (1, len(self.words)),  # y只有1个字，维度是词典数量
                dtype = np.bool
            )
            y_vec[0, self.word2numF(y)] = 1.0  # 找到这次对应的下标，从而在对应下标赋值~

            x_vec = np.zeros(
                # 注意这里X的shape与model的Input的shape是一样的  （6 , 词典大小）
                shape = (1, self.config.max_len, len(self.words)),  # x是三维，1- 跨度- 维度
                dtype = np.bool
            )
            for t, char in enumerate(x):
                x_vec[0, t, self.word2numF(char)] = 1.0

            yield x_vec, y_vec
            i+=1

    # keras建立模型
    def build_model(self):
        """
        使用keras来构建网络模型，使用GRU
        层是：输出层Input + Embedding + LSTM + Dropout + LSTM + Dropout + Flatten->Dense->softmax
        层优化：输入层Input + LSTM + drooput + LSTM + Drouput + Dense->softmax
        激活函数：softmax
        优化器：Adam
        :return:
        """
        print('building model')

        # 一个输入节点x的形状是 (6, 词典大小)，因为是one-hot编码吧，所以每次6个词，然后维度是词典大小(5552个词大概)
        # 直接把LSTM作为输入层第一层的话，要指定输入大小；当然如果使用变长序列，那么，只需要在LSTM层前加一个Masking层，或者embedding层即可。
        input_tensor = Input(shape=(self.config.max_len, len(self.words)))  # 输入的dimension，不包括batchsize,len(self.words)是5551，因为包括__len__，不然5550
        # ？LSTM中return_sequence，在输出序列中返回单个hiddenstate还是返回全部tiemstep的hiddenstate值，控制hidden_state，True输出全部，False输出最后一个
        lstm = LSTM(512, return_sequences=True )(input_tensor)  # 输出512维
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(256)(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(len(self.words), activation='softmax')(dropout)

        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        # input_tensor = Input(shape=(self.config.max_len, ) ))  # 输入的dimension，不包括batchsize
        # embedd = Embedding(len(self.num2word) + 1, 300, input_length=self.config.max_len)(input_tensor)
        # lstm = Bidirectional(GRU(128, return_sequences=True))(embedd)  # Bidirectional是RNN的双向封装器，对序列进行前向和后向计算
        # dropout = Dropout(0.6)(lstm)
        # lstm = Bidirectional(GRU(128, return_sequences=True))(embedd)
        # dropout = Dropout(0.6)(lstm)
        # flatten = Flatten()(lstm)
        # dense = Dense(len(self.words), activation='softmax')(flatten)

        # self.model = Model(inputs=input_tensor, outputs=dense)
        # optimizer = Adam(lr=self.config.learning_rate)
        # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 训练模型
    def train(self):
        # number_of_epoch = len(self.words) // self.config.batch_size  # 迭代次数  # 反正我不知道怎么算的&……
        number_of_epoch = len(self.files_content) - (self.config.max_len + 1) * self.poems_num  # 得到总共要移动这么多
        number_of_epoch /= self.config.batch_size  # 除以批量大小正好的迭代次数  # 除以批次
        number_of_epoch = int(number_of_epoch / 1.5)  # 迭代次数再减少

        print('epoches = ', number_of_epoch)  # 34858
        print('poems_num = ', self.poems_num) # 24027
        print('len(self.files_content) = ', len(self.files_content))


        if not self.model:
            self.build_model()

        self.model.fit_generator(
            generator = self.data_generator(),
            verbose=False,  # 日志显示
            steps_per_epoch = self.config.batch_size,  # 每个epoch中调用几次generator生成数据进行训练（也就是批量）
            epochs = number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)  # 每次epoch结束的时候，调用生成结果展示
            ]
        )

    def _preds(self, sentence, length=23, temperature=1):
        """
        供内部调用，输入sentence，sentence是maxlen长度的字符串，返回所需length长度的预测值字符串
        :param sentence: 要进行预测的输入值
        :param length:   预测后输出的字符长度
        :param temperature:
        :return: 返回生成length长度的句子
        """
        sentence = sentence[:self.config.max_len]  # 取前maxlen个字符
        generate = ''
        for i in range(length):
            pred = self._pred(sentence, temperature)  # 预测下一个字符
            generate += pred
            sentence = sentence[1:] + pred  # 保持前maxlen个字符的输入，窗口大小是6，移动1。从而保证每次都maxlen个输入
        return generate

    def _pred(self, sentence, temperature=1):
        """
        内部使用方法，根据一串输入，返回单个预测字符
        :param sentence:     前maxlen个字符
        :param temperature:
        :return:
        """
        # 小于maxlen情况
        if len(sentence) < self.config.max_len:
            print('in def _pred, length error')
            return
        # 大于maxlen情况
        sentence = sentence[-self.config.max_len:]  # 每次都要处理一下啊

        x_pred = np.zeros((1, self.config.max_len, len(self.words)+1))
        for t, char in enumerate(sentence):
            x_pred[0, t, self.word2numF(char)] = 1.  # 用one-hot标示

        preds = self.model.predict(x_pred, verbose=0)[0]  # 调用keras的predict预测下一个词

        next_index = self.sample(preds, temperature=temperature)  # 获取到下一个字符的下标
        next_char = self.num2word[next_index]  # 获取下一个字符串

        return  next_char

    # 预测模型 - 弃用
    """
        def predict(self, text):

        # 根据给出的文字，生成诗句，如果给的text不到四个字，则随机补全
        # :param text:
        # :return:

        if not self.loaded_model:
            return
        with open(self.config.poetry_file, 'r', encoding='utf-8') as f:
            file_list = f.readlines()
        random_line = random.choice(file_list)  # 随机选择一行

        # 如果给的text不到四个字，随机从字典里选缺的字来补全
        if not text or len(text) != 4:
            for _ in range(4-len(text)):
                random_str_index = random.randrange(0, len(self.words))
                text += self.num2word.get(random_str_index) \
                    if self.num2word.get(random_str_index) not in [',', '。', '，'] \
                    else self.num2word.get(random_str_index + 1)
        seed = random_line[-(self.config.max_len): -1]  # 得到随机选择的这行的最后导数几个字，留倒数第一个字

        res = ''  # 存放返回结果

        seed = 'c' + seed  # 给seed加个开头

        for c in text:
            seed = seed[1:] + c
            for j in range(5):  # 5个数
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(seed):
                    x_pred[0,t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, 1.0)  # 放的是一系列候选词？
                next_char = self.num2word[next_index]
                seed = seed[1:] + next_char
            res += seed
        return res
    """

    # 模型单个字符的输出结果
    def sample(self, preds, temperature=1.0):
        """
        训练过程中的每个epoch迭代中采样
        temperature = 1.0， 模型输出正常
        temperature = 0.5， 模型输出比较open
        temperature = 1.5， 模型输出比较保守
        训练过程可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
        sample就是吧一个predict得到的可能性数组，转为一定概率的分布后，随机选择概率较大的下标吧，得到下一个字

        :param preds:
        :param temperature:
        :return:
        """
        preds = np.asarray(preds).astype('float64')  # 转换为numpy数组再用浮点数标示
        # preds = np.log(preds) / temperature          #
        # exp_preds = np.exp(preds)
        exp_preds = np.power(preds, 1./temperature)  # 设置输出的权值，从而保守or开放
        preds = exp_preds / np.sum(exp_preds)        # 必须这样标准化,类似softmax输出的看概率，标准化后才可以用random.choice
        # numpy.random.multinomial(n, pvals, size=None), n是实验次数，pvals是长度序列，表示每次概率；size是
        # probas = np.random.multinomial(1, preds, 1)  # multinomial是多项式分布采样
        # return np.argmax(probas)  # 选择概率最大的下标，也就是对应的词

        prob = np.random.choice(range(len(preds)), 1, p=preds)  # 从range(len(preds))中，以概率p，随机选择1个，抽样之后还放回去, 返回一个list
        return int(prob.squeeze())  # squeeze()从数组的形状中删除单维度条目，也就是吧shape为1的维度去掉, 并转为int

    # 生成训练结果，每个epoch打印出当前学习情况
    def generate_sample_result(self, epoch, logs):
        """
        改成每4个epoch打印出当前学习情况
        :param epoch:
        :param logs:
        :return:
        """
        if epoch % 10 != 0:
            return

        with open('out.txt', 'a', encoding='utf-8') as f: # 追加写入
            f.write('=================Epoch {} ================\n'.format(epoch))

        # print('\n=================Epoch {} ================'.format(epoch))
        for diversity in [0.5, 1.0, 1.5]:
            print('--------------Diversity {} -------------'.format(diversity))
            generate = self.predict_random(temperature = diversity)  # 生成一首诗
            print(generate)

            # 训练时的预测结果写入text
            with open('out.txt', 'a', encoding='utf-8')as f:
                f.write(generate + '\n')

            # start_index = random.randint(0, len(self.files_content) - self.config.max_len - 1)
            # generated = ''
            # sentence = self.files_content[start_index: start_index + self.config.max_len]
            # generated += sentence
            # for i in range(20):
            #
            #     x_pred = np.zeros((1, self.config.max_len))
            #     for t, char in enumerate(sentence[-6:]):
            #         x_pred[0,t] = self.word2numF(char)
            #
            #     preds = self.model.predict(x_pred, verbose=0)[0]
            #     next_index = self.sample(preds, diversity)
            #     next_char = self.num2word[next_index]
            #
            #     generated += next_char
            #     sentence += next_char
            # print(sentence)

    # 随机生成一首诗
    def predict_random(self, temperature=1):
        """
        随机从库里选取一句开头的诗句，生成五言绝句！
        :param temperature:
        :return:
        """
        if not self.model:
            print('model not loaded')
            return

        index = random.randint(0, self.poems_num)  # 选取某篇诗
        sentence = self.poems[index][:self.config.max_len]              # 某篇诗第一句（也就是前max_len个字）
        generate = self.predict_sen(sentence, temperature=temperature)  # 根据第一句，产生后面的句子
        return generate

    # 给定第一句，开始作诗
    def predict_sen(self, text, temperature=1):
        """
        根据前max_len个字，生成诗句，此例中，即根据给出的第一句诗句（含逗号），来生成古诗
        :param sentence:
        :param temperature:
        :return:
        """
        if not self.model:
            return

        max_len = self.config.max_len
        if len(text) < max_len:
            print('length should not be less that ', max_len)
            return
        sentence = text[-max_len:]  # 得到传入句子的最后maxlen个字
        print('the first line:', sentence)
        generate = str(sentence)    # 转为字符串
        generate += self._preds(sentence, length = 24-max_len, temperature=temperature)
        return generate

    # 藏头诗
    def predict_hide(self, text, temperature=1):
        """
        给前面4个字，生成藏头五言绝句
        :param text:
        :param temperature:
        :return:
        """
        if not self.model:
            print('model not loaded')
            return
        if len(text) != 4:
            print('藏头诗输入必须是4个字！')
            return

        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len-1字符 + 给出的首个文字作为初始输入，从而开始生成藏头诗的第二个字 乃至第一句
        sentence = self.poems[index][ 1-self.config.max_len : ] + text[0]  # 所以有一句话+6个字
        generate = str(text[0])
        # print('first line = ', sentence)
        # 第一句话处理：生成一句话（包括符号）
        for i in range(5):
            next_char = self._pred(sentence, temperature)
            sentence = sentence[1:] + next_char  # 相当于每次向右滑动1个
            generate += next_char

        for i in range(3):
            generate += text[i+1]
            sentence = sentence[1:] + text[i+1]
            for i in range(5):
                next_char = self._pred(sentence, temperature)
                sentence = sentence[1:] + next_char
                generate += next_char

        return generate

    # 给第一个字，开始作诗
    def predict_first(self, char, temperature=1):

        if not self.model:
            print('model not loaded')
            return

        # 随机选一首诗的最后maxlen-1长度的字+text，作为初始输入
        index = random.randint(0, self.poems_num)
        sentence = self.poems[index][1-self.config.max_len:] + char
        print('first line = ', sentence)
        generate = str(char)

        # 直接预测后面23个字符
        generate += self._preds(sentence, length=23, temperature=temperature)
        return generate



# 运行
if __name__ == '__main__':

    from config import Config   # 需要了再导入
    model = PoetryModel(Config)
    print('模型加载完成')

    # 给出第一句话进行预测
    for i in range(3):
        sen = model.predict_sen('床前明月光，')
        print(sen)



    print('\n藏头诗')
    # 藏头诗
    # text = input('输入藏头诗前4个字：')
    text = '福州大学'
    for i in range(3):
        sen = model.predict_hide(text)
        print(sen)



    print('\n给出一个字进行预测')
    # 给出第一个字进行预测
    # text = input('请输入第一个字：')
    text = '美'
    for i in range(3):
        sen = model.predict_first(text)
        print(sen)



    # 随机抽取第一句话进行预测
    print('\n随机选一首诗进行预测')
    for temp  in [0.5, 1.0, 1.5]:
        sen = model.predict_random(temperature= temp)
        print(sen)