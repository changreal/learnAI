# 引入包
import random
import jieba
import pandas as pd
import numpy as np

# 加载文件们
stopwords = pd.read_csv('stopwords.txt', index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values

laogong_df = pd.read_csv('beilaogongda.csv', encoding='utf-8', sep=',')
laopo_df = pd.read_csv('beilaogongda.csv', encoding='utf-8', sep=',')
erzi_df = pd.read_csv('beierzida.csv', encoding='utf-8', sep=',')
nver_df = pd.read_csv('beinverda.csv', encoding='utf-8', sep=',')

# 预处理
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)

laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()

def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]  # 去数字
            segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
            segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
            segs = list(filter(lambda x: x not in stopwords, segs))  # 去掉停用词
            sentences.append((" ".join(segs), category))  # 打标签
        except Exception:
            print(line)
            continue

sentences = []
preprocess_text(laogong, sentences, 0)
preprocess_text(laopo, sentences, 1)
preprocess_text(erzi, sentences, 2)
preprocess_text(nver, sentences, 3)

# 打散与生成数据
random.shuffle(sentences)
for sentence in sentences[:10]:
    print(sentence[0], sentence[1])  # 输出打印数据看看
all_texts = [sentence[0] for sentence in sentences]  # 获取文本list- 1维  ['报警 女儿 打伤 无需 民警 到场', ...]
all_labels = [sentence[1] for sentence in sentences] # 获取标签list

"""
开始用keras的KSTM进行分类
"""
# 开始使用keras的LSTM对数据进行分类！
# 引入keras所需模块
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Embedding, GRU
from keras.layers import Dropout, Dense

"""
keras预处理
"""

# 预定义一些变量
MAX_SEQUENCE_LENGTH = 100  # 序列填充时最大序列长度
EMBEDDING_DIM       = 200  # 词嵌入维度
VALIDATION_SPLIT    = 0.15 # 验证集比例
TEST_SPLIT          = 0.2  # 测试集比例

# keras的sequence模块文本序列填充
tokenizer  = Tokenizer()                             # keras分词器
tokenizer.fit_on_texts(all_texts)                    # 要用来训练的文本序列化，后面的步骤基于它
# sequences 和 word_index对应
sequences  = tokenizer.texts_to_sequences(all_texts) # 文本序列化（整数编码，因为后面Embedding层要求对输入数据整数编码），变为list,[3,7,17,5,1,2],[]...[]]
word_index = tokenizer.word_index                    # 词典 及 下标, dict, {'民警': 1, '到场': 2, '报警': 3, '老公': 4, '无需': 5, '持械': 6,..}
print('Found %s unique toknes.' % len(word_index))   # 输出分词数量  391个

# keras只接受长度相同序列输入，如果序列长度不同，需要pad_sequences将序列转化为经过填充以后的一个长度相同的新序列
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 序列填充，填充完后后面才可以训练
labels = to_categorical(np.asarray(all_labels))              # 类别标签one-hot编码
print('Shape of data tensor:', data.shape)  # (1722,100) 100维，是序列化后填充后的维度 n * k
print('Shape of label tensor:', labels.shape) #(1722, 4) 4维

"""
数据切分
"""
# 数据切分（上面已经打乱数据了shuffle了）
p1 = int(len(data)*(1 - VALIDATION_SPLIT - TEST_SPLIT))  # len(data)==1172
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val   = data[p1:p2]
y_val   = labels[p1:p2]
x_test  = data[p2:]
y_test  = labels[p2:]

"""
LSTM模型训练
"""
model = Sequential()
# 词嵌入层是：输出：词汇表大小*词嵌入维度的矩阵； 输入：输入序列的长度
word_embedding = Embedding(len(word_index)+1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)
model.add(word_embedding)
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))  # 200输出维度，drooput第一个是x hidden之间的，第二个droupout是hidden-hidden之间的
model.add(Dropout(0.2))  # (层与层之间的dropout)训练过程中每次更新参数时随机断开一定比例0.2的输入神经连接，用于防止过拟合
model.add(Dense(64, activation='relu'))  # 输出64维， 激活函数relu
model.add(Dense(labels.shape[1], activation='softmax'))  # units是代表该层输出维度， output=activation(dot(input, kernel)+bias),
model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy',  # 对于多分类问题，可以用分类交叉熵（categorical crossentropy）损失函数；输出概率值的模型，交叉熵往往是最好的选择。
              optimizer='rmsprop', # 大多数情况下，使用 rmsprop 及其默认的学习率是稳妥的。
              metrics=['acc']
              )
print(model.metrics_names)  #['loss', 'acc']

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)
model.save('lstm.h5')

# 评估
print(model.evaluate(x_test, y_test))  # [0.3794687170913254, 0.7391304368558137]

"""
用GRU模型进行文本分类（值只需要改变模型训练部分）
"""
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(GRU(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
               optimizer='rmsprop',
               metrics=['acc']
              )
print(model.metrics_names)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)
model.save('GRU.h5')
print(model.evaluate(x_test, y_test))
