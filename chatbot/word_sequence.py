# -*- coding:utf-8 -*-

import numpy as np

#  词与句子（字典、统计、词与向量的转化）
class WordSequence(object):
    PAD_TAG = '<pad>'  # 填充标签，补位
    UNK_TAG = '<unk>'  # 未知标签
    START_TAG = '<s>'  # 开始标记
    END_TAG = '</s>'  # 结束标记

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    # 字典初始化
    def __init__(self):
        # 初始化基本字典dict
        self.fited = False
        self.dict = {
            # 这些是字典的打头标记，打头标记完是词们
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END
        }

    # 词换成index
    def to_index(self, word):
        assert self.fited, 'WordSequence尚未进行fit操作'  # 用一个assert语句来测试condition
        if word in self.dict:  # 词在字典里，
            return self.dict[word]  # 返回位置
        return WordSequence.UNK  # 不在的话返回unknow，这里unk的词频用1来表示

    # index换成word
    def to_word(self, index):
        assert self.fited, 'WordSequence尚未进行fit操作'
        for k, v in self.dict.items():  # 这样每次都要跟字典里的值匹配比较，时间复杂度和字典大小成正比；dict:key-value
            if v == index:
                return k
        return WordSequence.UNK_TAG

    # 字典大小
    def size(self):
        assert self.fited, 'WordSequence尚未进行fit操作'
        return len(self.dict) + 1  # 加1是为了做长度的补位

    # 定义长度（继承方法）
    def __len__(self):
        return self.size()

    # 训练函数（来训练字典）
    # 参数：句子、最小出现次数、最大出现次数、最大特征数(因为要提取特征)
    def fit(self, sentences, min_count=5, max_count=None, max_features=None):

        assert not self.fited, 'WordSequence只能fit一次'

        count = {}  # 词频
        for sentence in sentences:  # 传进句子进来，然后开始简单统计
            arr = list(sentence)
            for a in arr:
                if a not in count:
                    count[a] = 0
                count[a] += 1
        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}  # 只有大于最小词频的词统计才有意义，否则统计词典不收录
        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}  # 同样，小于最大词频的词统计才有意义

        # 字典简单初始化
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END
        }

        if isinstance(max_features, int):  # 如果features是int的话
            count = sorted(list(count.items()), key=lambda x: x[1])  # 用items排序
            # 如果没有限制features的长度（max_features）则用所有的，
            # 如果限制了，取词频最高的max_features长的特征。
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]  # 将count缩小到在最大特征值范畴内的字典
            for w, _ in count:  # 用不上第二个参数
                self.dict[w] = len(self.dict)  # 不断赋予新的下标
        else:
            for w in sorted(count.keys()):  # 用keys排序
                self.dict[w] = len(self.dict)  # 不断赋予新的下标

        self.fited = True  # 训练标志，一开始False默认没有训练

    # 句子转换为向量，向量再转换为句子
    # 参数：句子，句子最大长度
    def transform(self, sentence, max_len=None):
        assert self.fited, 'WordSequence尚未进行fit操作'

        if max_len is not None:
            r = [self.PAD] * max_len  # 用到填充位置，进行填充，比如最大长度4,则r = [0,0,0,0]
        else:
            r = [self.PAD] * len(sentence)  # 即r不然是最大长度、不然是句子长度

        for index, a in enumerate(sentence):  # 枚举句子,index是下标，a是词
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(a)  # 每个词转化为index（字典下标）

        return np.array(r)

    # 向量转换为句子
    # 参数：一段为向量的句子、是否忽略填充位、是否忽略unk类型、是否忽略start类型
    def inverse_transform(self, indices, ignore_pad=False, ignore_unk=False, ignore_start=False, ignore_end=False):
        ret = []
        for i in indices:
            word = self.to_word(i)  # 数字转换为字
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and ignore_end:
                continue

            ret.append(word)  # []要append

        return ret


def test():
    ws = WordSequence()
    ws.fit([['你', '好', '啊'], ['你', '好', '哦'],['你','不','啊'],['你','桑','师傅'] ],3) # 得到词典
    # print(len(ws))  # 6

    indice = ws.transform(['我', '们', '好','你'])
    print(indice)  # 因为没做hash，所以是这些符号

    back = ws.inverse_transform(indice)
    print(back)


if __name__ == '__main__':
    test()
