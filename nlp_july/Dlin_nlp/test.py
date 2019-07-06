import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# 一个小例子，看看日常生活中用RNN怎么玩的，一个小例子
# 字母级别

with open('../input/Winston_Churchil.txt',encoding='utf-8') as f:
    raw_text = f.read()
raw_text = raw_text.lower()


# 既然我们是以每个字母为层级，字母总共才26个，所以我们可以很方便的用One-Hot来编码出所有的字母（当然，可能还有些标点符号和其他noise）
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# 构造训练集
seq_length = 100  # 序列长度，时间步伐
x = []  # x是前置字母们，长度序列为100
y = []  # y是后置字母们，记录此序列结束时的下标—
for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append( [char_to_int[char] for char in given])
    y.append(char_to_int[predict])

# 数据处理——为了套LSTM
#我们已经有了一个input的数字表达（index），我们要把它变成LSTM需要的数组格式： [样本数，时间步伐，特征]
# 第二，对于output，我们在Word2Vec里学过，用one-hot做output的预测可以给我们更好的效果，相对于直接预测一个准确的y数值的话。
n_patterns = len(x)
n_vocab = len(chars)

#把x变成LSTM需要的样子
x = numpy.reshape(x, (n_patterns, seq_length, 1))  # 调整形状
x = x / float(n_vocab)  # 简单normal到0-1之间
# output变成one-hot， 编码输出便签，多分类标签
# （当类别比较多的时候输出值的跨度就会比较大，此时输出层的激活函数就只能用linear
y = np_utils.to_categorical(y)

# print(x[11])
# print(y[11])


# LSTM模型建造
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2] )))  # 100,1
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x, y, nb_epoch=50, batch_size=4096) # 跑不动的

# 看看效果
def predict_next(input_array):
    x = numpy.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input)-seq_length):]:
        res.append(char_to_int[c])
    return res

def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c

def generate_article(init, rounds=200):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string

init = 'His object in coming to New York was to engage officers for that service. He came at an opportune moment'
article = generate_article(init)
print(article)