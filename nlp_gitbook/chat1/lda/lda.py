import gensim

# #引入库文件
import jieba.analyse as analyse
import jieba
import pandas as pd
from gensim import corpora, models, similarities
import gensim
import numpy as np
import matplotlib.pyplot as plt

#设置文件路径与加载语料、停用词
dir = "./"
file_desc = "".join([dir,'car.csv'])
stop_words = "".join([dir,'stopwords.txt'])

df = pd.read_csv(file_desc, encoding='gbk')
stopwords=pd.read_csv(stop_words,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')  # 指定列名
stopwords=stopwords['stopword'].values  # .column.values以array形式返回指定column的所有取值  array([,,,])

# 预处理
df.dropna(inplace = True)          # 删除nan行
lines = df.content.values.tolist()  # 取内容转为list

# 开始分词
sentences=[]
for line in lines:
    try:
        segs=jieba.lcut(line)
        segs = [v for v in segs if not str(v).isdigit()]#去数字
        segs = list(filter(lambda x:x.strip(), segs))   #去左右空格
        segs = list(filter(lambda x:x not in stopwords, segs)) #去掉停用词
        sentences.append(segs)
    except Exception:
        print(line)
        continue

# 构建词袋模型
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]  # .doc2bow是把文档doc变成一个稀疏向量，词袋模型;list 形如[[(1,2),(4,1)],[]]
# lda模型
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)  # num_topics是主题个数，这里定义10个
print(lda.print_topic(3, topn=5))  # 查看第1号分类，最长出现的5个词
for topic in lda.print_topics(num_topics=10, num_words=8):
    print(topic)  # 打印所有10个主题， 每个主题显示8个词

# 中文matplotlib
plt.rcParams['font.sans-serif'] = [u'SimHei']  # 这个是为了正常显示中文
plt.rcParams['axes.unicode_minus'] = False

# 在可视化部分，我们首先画出了九个主题的7个词的概率分布图
# 不太会画图
num_show_term = 8 # 每个主题下显示几个词
num_topics  = 10
for i, k in enumerate(range(num_topics)):
    ax = plt.subplot(2, 5, i+1)                        # 2行5列 第i个图形
    item_dis_all = lda.get_topic_terms(topicid=k)      # 获取主题
    item_dis = np.array(item_dis_all[:num_show_term])  # 前8个词
    ax.plot(range(num_show_term), item_dis[:, 1], 'b*')
    item_word_id = item_dis[:, 0].astype(np.int)
    word = [dictionary.id2token[i] for i in item_word_id]
    ax.set_ylabel(u"概率")
    for j in range(num_show_term):
        ax.text(j, item_dis[j, 1], word[j], bbox=dict(facecolor='green',alpha=0.1))
plt.suptitle(u'9个主题及其7个主要词的概率', fontsize=18)
plt.show()
