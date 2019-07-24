import random
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import multiprocessing


"""
加载与初步处理
"""
# 加载停用词
stopwords = pd.read_csv('stopwords.txt', index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values
# 加载语料与处理
laogong_df = pd.read_csv('beilaogongda.csv', encoding='utf-8', sep=',')
laopo_df = pd.read_csv('beilaogongda.csv', encoding='utf-8', sep=',')
erzi_df = pd.read_csv('beierzida.csv', encoding='utf-8', sep=',')
nver_df = pd.read_csv('beinverda.csv', encoding='utf-8', sep=',')
# 删除语料的nan行
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)
# 转换
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()

"""
预处理 分词与去停用词，就不用打标签了 其他差不多
"""
def preprocess_text(content_lines, sentences):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]  # 去数字
            segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
            segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
            segs = list(filter(lambda x: x not in stopwords, segs))  # 去掉停用词
            sentences.append(" ".join(segs))
        except Exception:
            print(line)
            continue

sentences = []
preprocess_text(laogong, sentences)
preprocess_text(laopo, sentences)
preprocess_text(erzi, sentences)
preprocess_text(nver, sentences)

random.shuffle(sentences)

for sentence in sentences[:10]:
    print(sentence)


"""
抽取词向量特征
"""
# tf-idf方式
# TfidfTransformer + CountVectorizer  =  TfidfVectorizer
# 应用线性缩放tf，例如1+log(tf)覆盖tf
# max_df 这个给定特征可以应用在 tf-idf 矩阵中，用以描述单词在文档中的最高出现率。假设一个词（term）在 80% 的文档中都出现过了，那它也许（在剧情简介的语境里）只携带非常少信息。
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5) # 将文本中的词语转换为词频矩阵，矩阵元素a[i][j]表示j词在i类文本下的词频
transformer = TfidfTransformer()  # 统计每个词语的tf-idf权值
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的td-idf权重
print('features length:' + str(len(word)))  # 查看特征大小
print(tfidf)

"""
tf-idf的中文文本k-means聚类
"""
# 使用k-means++初始化模型，当然也可以随机初始化，然后通过PCA降维把权重weight降到10维，然后聚类模型训练
numClass = 4
clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  ##这里也可以选择随机初始化init="random"
pca = PCA(n_components=10) # 降到10维
TnewData = pca.fit_transform(weight) # 载入N维, 吧之前tf-idf矩阵用pca降维到10维
s = clf.fit(TnewData)  # 降维后的数据放到k-means训练

# 定义聚类结果可视化函数plot_cluster
def plot_cluster(result, newData, numClass):
    """
    绘制代码第一部分绘制结果 newData，第二部分绘制聚类的中心点：
    :param result:   聚类拟合的结果集(就是之前kmeans模型拟合的数据，放在list)
    :param newData:  权重weight降维的结果，这里需要降维到2维，即平面可视化
    :param numClass: 聚类分为几簇
    :return:
    """
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index+=1  # [[0],[1],...[n]]
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^', 'g^'] * 3




# 把数据姜降到2维，然后获得结果，绘制聚类结果图
pca = PCA(n_components=2)  # 输出2维
newData = pca.fit_transform(weight) #  wegith姜维到2维
result = list(clf.predict(TnewData)) # clf是kmeans
plot_cluster(result, newData, numClass)

