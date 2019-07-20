    # -*- coding:utf-8 -*-

import random
import numpy as np
from tensorflow.python.client import device_lib
from word_sequence import  WordSequence

'''
定义工具类
'''

VOCAB_SIZE_THRESHOLD_CPU = 50000


def _get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# 根据输入输出的字典大小，选择在CPU还是GPU上执行Embedding
def _get_embed_device(vocab_size):
    gpus = _get_available_gpus()
    if not gpus or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    return "/gpu:0"

# 单独句子转换
def transform_sentence(sentence, ws, max_len=None, add_end=False):
    encoded = ws.transform(sentence, max_len=max_len if max_len is not None else len(sentence))   # 返回句子转换成向量后的结果
    encoded_len = len(sentence) + (1 if add_end else 0)
    if encoded_len > len(encoded):
        encoded_len = len(encoded)
        # [4,5,6,7], 它的长度
    return encoded, encoded_len   # encode是一个转换后的数组，以及长度


# 从数据中随机生成batch_size的数据，然后给转换后输出
def batch_flow(data, ws, batch_size, raw=False, add_end=True):
    # 输入：data数组(要包含1个or多个同等数据队列的数组)，
    # ws数量和data数量保持一致，len(ws)==len(data)， ws里包含字典和添加位
    # raw是是否返回原始对象，如果是true，假设结果ret，那么len(ret)==len(data)*3，如果是false，那么len(ret)==len(data)*2
    # 比如要输入问题队列和答案队列，那么
        # 问题队列是 Q = (Q1,Q2,....QN)，
        # 答案队列是 A = (A1,A2,...AN),     len(Q)==len(A)
    # 是否返回原始对象，是否添加原始标记
    # data要转换，变成list 然后去拿所有数据，
    """
    Q = (q1, q2, q3 ..., qn)
    A = (a1, a2, a3 ..., an)
    len(Q) == len(A)
    batch_flow([Q,A], ws, batch_size = 32)
    raw = False:
    next(generator) == q_i_encoded, q_i_len, a_i_encoded, a_i_len   # 返回的格式
    raw = True:
    next(generator) == q_i_encoded, q_i_len, a_i_encoded, a_i_len, a_i  # 返回的格式

    :param data:
    :param ws:
    :param batch_size:
    :param raw: 是否返回原始对象，如果是True，假设结果是ret，那么len(ret) == len(data)*3
    如果是False，那么len(ret) == len(data) * 2
    :param add_end:
    :return:
    """
    all_data = list(zip(*data))  # [(['a','b'],[1,2])]形式 变成每个 word-频率对应
    if isinstance(ws, (list, tuple)):   # 数据类型判断，list or tuple
        assert len(ws) == len(data), 'ws的长度必须等于data的长度'   # 找出数据长度  ws长度是2，data长度也是2 因为传入的时候分装成了3维数组，第一维是2
    if isinstance(add_end, bool):       # 添加结束标记是否是布尔值
        add_end = [add_end] * len(data) # 形如[True, True]
    else:
        assert(isinstance(add_end, (list, tuple))), 'add_end不是boolean，应该是一个list(tuple) of boolean'
        assert len(add_end) == len(data), '如果add_end是list(tuple),那么add_end长度应该和输入数据的长度一样'
    mul = 2
    if raw:
        mul = 3

    while True:
        data_batch = random.sample(all_data, batch_size)
        batches = [[] for i in range(len(data) * mul)]  # 数组,返回结果是 [[],[],[],...,[]]  二维数组，长度是len*mul

        max_lens = []
        for j in range(len(data)):   # data长度
            # # hasattr 是否包含__len__属性，如果不包含返回0，
            # + 结尾标记是补位操作
            # 取出当前句子的最大长度
            max_len = max([len(x[j]) if hasattr(x[j], '__len__') else 0 for x in data_batch])  + (1 if add_end[j] else 0)
            max_lens.append(max_len)   # 就是当前批量中有最大长度词典的那个

        for d in data_batch:            # d是每个批量里的数据对循环，d是一篇文档   循环批量  tuple形式(['a','bb'],1,3,3,4)
            for j in range(len(data)):  # 词 and  target                        循环数据 和 标签
                if isinstance(ws, (list, tuple)):
                    w = ws[j]   # 取出行里的最大数据
                else:
                    w = ws

                # 添加结尾标记
                line = d[j]  #
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]
                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])  # 做一个句子转换
                    batches[j * mul].append(x)    # 添加x-[],进来, 长度是j*mul长度
                    batches[j * mul + 1].append(xl)   # 尾部添加len长度
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)
                if raw:   # 最后操作的时候，乘的数要+1
                    batches[j * mul + 2].append(line)

        batches = [np.asarray(x) for x in batches]    # batches是为了找到 [q, qlen, a, alen]，将结构数据转化为ndarray
        yield batches  # 产生批量数据

# 实际中随机做切分数据方式 bucket
def batch_flow_bucket(data, ws, batch_size, raw=False, add_end=True, n_bucket=5, buket_ind=1, debug=False):
    """
    :param data:
    :param ws:
    :param batch_size:
    :param raw:
    :param add_end:
    :param n_bucket:  生成多少个bucket,如果n==1实际上相当于没有
    :param buket_ind: 是指哪一个维度的输入作为bucket的依据
    :param debug:
    :return:
    """
    all_data = list(zip(*data))
    lengths = sorted(list(set([len(x[buket_ind]) for x in all_data])))  # 传入的数据维度的，长度排序
    if n_bucket > len(lengths):
        n_bucket = len(lengths)
    splits = np.array(lengths)[  # 切分
        (np.linspace(0, 1, 5, endpoint=False) * len(lengths)).astype(int)
    ].tolist()   # 等差切分，再转换成list, list里都是整数
    splits += [np.inf]  # 结尾用无限大函数 np.inf表示无限大的正整数
    if debug:
        print(splits)

    # 数据操作
    ind_data = {}   # 此维度上的数据集合
    for x in all_data: # 在数据对上进行
        l = len(x[buket_ind])  # 取数据长度
        for ind, s in enumerate(splits[:-1]):
            if s <= l <= splits[ind + 1]:
                if ind not in ind_data:
                    ind_data[ind] = []
                ind_data[ind].append(x)
                break
    inds = sorted(list(ind_data.keys()))
    ind_p = [len(ind_data[x]) / len(all_data) for x in inds]
    if debug:
        print(np.sum(ind_p), ind_p)
    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), 'len(ws)必须等于len(data)'
    if isinstance(add_end, bool):
        add_end = [add_end] * len(data)
    else:
        assert (isinstance(add_end, (list, tuple))), 'add_end不是boolean'
        assert len(add_end) == len(data), '如果add_end是list(tuple),那么add_end长度应该和输入数据的长度一样'
    mul = 2
    if raw:
        mul = 3

    while True:
        choice_ind = np.random.choice(inds, p=ind_p)
        if debug:
            print('choice_ind', choice_ind)

        data_batch = random.sample(ind_data[choice_ind], batch_size)
        batches = [[] for i in range(len(data) * mul)]
        max_lens = []
        for j in range(len(data)):
            max_len = max([len(x[j]) if hasattr(x[j], '__len__') else 0 for x in data_batch]) \
                      + (1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws

                line = d[j]
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]
                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul + 1].append(xl)
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)
                if raw:
                    batches[j * mul + 2].append(line)

        batches = [np.asarray(x) for x in batches]
        yield batches


def test_batch_flow():
    from fake_data import generate
    # 生成所需要的数据！ 10000条句子，
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow([x_data, y_data], [ws_input, ws_target], 4)
    x, xl, y, yl = next(flow)
    print(x.shape, y.shape, xl.shape, yl.shape)


def test_batch_flow_bucket():
    from fake_data import generate
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow_bucket([x_data, y_data], [ws_input, ws_target], 4, debug=True)

    for _ in range(10):
        x, xl, y, yl = next(flow)
        print(x.shape, y.shape, xl.shape, yl.shape)


if __name__ == '__main__':
    # test_batch_flow()
    test_batch_flow_bucket()
    # size = 300000
    # # print(_get_available_gpus())
    # print( _get_embed_device(size))
