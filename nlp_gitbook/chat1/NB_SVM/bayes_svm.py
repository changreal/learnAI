# 引入包
import random
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

dir = './'
# 指定语料文件提取
stop_words = "".join([dir, 'stopwords.txt'])
laogong = "".join([dir,'beilaogongda.csv'])  #被老公打
laopo = "".join([dir,'beilaopoda.csv'])  #被老婆打
erzi = "".join([dir,'beierzida.csv'])   #被儿子打
nver = "".join([dir,'beinverda.csv'])    #被女儿打
stopwords=pd.read_csv(stop_words,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values  #  返回一个array
#加载语料
laogong_df = pd.read_csv(laogong, encoding='utf-8', sep=',')
laopo_df = pd.read_csv(laopo, encoding='utf-8', sep=',')
erzi_df = pd.read_csv(erzi, encoding='utf-8', sep=',')
nver_df = pd.read_csv(nver, encoding='utf-8', sep=',')

# 预处理
laogong_df.dropna(inplace=True) # 删除nan
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)
laogong = laogong_df.segment.values.tolist()  # 转换为list
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()

#定义分词和打标签函数preprocess_text
# content_lines即为上面转换的list，sentences是定义的空list，用来储存打标签之后的数据
# 参数category 是类型标签
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]#去数字
            segs = list(filter(lambda x:x.strip(), segs))   #去左右空格
            segs = list(filter(lambda x:len(x)>1, segs)) #长度为1的字符
            segs = list(filter(lambda x:x not in stopwords, segs)) #去掉停用词
            sentences.append((" ".join(segs) , category))# 打标签，所以变成tuple('word word'  , 'laogong')形式
        except Exception:
            print(line)
            continue

# 生成训练数据以及切分
sentences = []
preprocess_text(laogong, sentences, 'laogong')
preprocess_text(laopo, sentences, 'laopo')
preprocess_text(erzi, sentences, 'erzi')
preprocess_text(nver, sentences, 'nver')
print(sentences)

random.shuffle(sentences)  # 打乱数据，生成可靠训练集

for sentence in sentences:
    print(sentence[0], sentence[1])  # 输出查看

x,y = zip(*sentences)  # 分开数据与标签
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)  # random_state就是保证每次都分割一样的训练集和测试集

# 抽取特征,对文本抽取词袋模型的特征
vec = CountVectorizer(
    analyzer='word',   # 可以设置为string\word\char\char_wb类型，这里设置为词袋模型特征抽取
    max_features=4000, # 取前4000个作为关键词集
)
vec.fit(x_train)  # 对训练集进行特征抽取

# 训练
classifier = MultinomialNB()  # 用朴素贝叶斯的多项式模型进行训练
classifier.fit(vec.transform(x_train), y_train)

# 对结果评估
print(classifier.score(vec.transform(x_test), y_test))

# 特征优化
vec = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4),  # 切分词组的长度范围
    max_features=20000  # 特征取高
)
vec.fit(x_train)
classifier.fit(vec.transform(x_train), y_train)
print(classifier.score(vec.transform(x_test), y_test))


# 用SVM来训练数据
svm = SVC(kernel='linear')
svm.fit(vec.transform(x_train), y_train)
print(svm.score(vec.transform(x_test), y_test))