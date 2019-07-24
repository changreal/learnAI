#-*- coding: utf-8 -*-
import random
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

dir = './data/'
stopwords=pd.read_csv(dir+'stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values

laogong_df = pd.read_csv(dir+'beilaogongda.csv', encoding='utf-8', sep=',')
laopo_df = pd.read_csv(dir+'beilaopoda.csv', encoding='utf-8', sep=',')
erzi_df = pd.read_csv(dir+'beierzida.csv', encoding='utf-8', sep=',')
nver_df = pd.read_csv(dir+'beinverda.csv', encoding='utf-8', sep=',')

"""
数据处理
"""
# 去掉nan行
laogong_df.dropna(inplace=True)    # 这个参数是指直接在原有的对象上面进行操作
laopo_df.dropna(inplace=True)      # 如果inplace为false那么会返回一个新的对象
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)
# 转换
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()

# 定义分词和打标签的预处理函数，也就是分词、去停、打标签
def prepocess_text(content_lines,  sentences, category):
    """
    定义分词和打标签的预处理函数
    :param content_lines: 为上面转换后的list
    :param sentences:     是定义的空list，用来存储打标签之后的数据
    :param category:      类型标签
    :return:
    """
    for line in content_lines:
        try:
            segs = jieba.lcut(line)  # 切出每句话分词的list列表
            segs = [v for v in segs if not str(v).isdigit()]        # 去除数字
            segs = list(filter(lambda x:x.strip(), segs))           # 去除左右空格
            segs = list(filter(lambda x:len(x)>1, segs))            # 筛选出长度大于1的词
            segs = list(filter(lambda x:x not in stopwords, segs))  # 筛选出非停用词
            sentences.append((" ".join(segs), category))            # 打标签
            # 会变为：sentences[('她被 老公 打了','0'), (), ...()]形式
        except Exception:
            print(line)
            continue

"""
打标签、生成数据
"""
sentences = []
prepocess_text(laogong,sentences, 0)
prepocess_text(laopo,sentences, 1)
prepocess_text(erzi,sentences, 2)
prepocess_text(nver,sentences, 3)
random.shuffle(sentences)  # 随机生成数据

# 打印出来康康
# for sentence in sentences[:10]:
#     print(sentence[0], sentence[1])


"""
抽取特征向量
"""
# 抽取特征，定义文本抽取词袋模型特征
vec = CountVectorizer(
    analyzer='word',
    max_features=4000
)
# 数据切分
x,y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)
# 训练数据转换为词袋模型
vec.fit(x_train)

"""
算法建模、模型训练
"""
classifier = MultinomialNB()  # 分类器用朴素贝叶斯的多项式模型
classifier.fit(vec.transform(x_train), y_train)

"""
评估、计算AUC值
"""
# 评估 计算AUC值
print('用多项式的朴素贝叶斯方式得分：',classifier.score(vec.transform(x_test), y_test))  # 用测试集评估
# 预测
pre = classifier.predict(vec.transform(x_test))         # 用测试集预测

"""
模型对比：
1. 改变特征向量模型
2. 改变训练模型
"""
# 改变特征向量模型：尝试加入抽取2-gram, 3-gram统计特征，把词库的量放大一点
vec = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4),  # ngram  2\3
    max_features=20000, # 最大特征量调大
)
vec.fit(x_train)
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)
print('用改进特征向量模型的多项式的朴素贝叶斯方式得分：',classifier.score(vec.transform(x_test), y_test))

# 改变训练模型
svm = SVC(kernel='linear')
svm.fit(vec.transform(x_train), y_train)
print('用SVM算法模型的得分：',svm.score(vec.transform(x_test), y_test))
# xgb矩阵赋值
xgb_train = xgb.DMatrix(vec.transform(x_train), label=y_train)
xgb_test = xgb.DMatrix(vec.transform(x_test))
