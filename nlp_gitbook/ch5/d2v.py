# doc2vec的输入是labeledSentence对象组成的迭代器，这个迭代次接收2个list作为初始化参数：word list 和 label list
# 把语料的每一句话当做一个文本，添加对应的标签。

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import jieba

# 定义停用词、标点符号
punctuation = [",", "。", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
sentences = [
    "长江是中国第一大河，干流全长6397公里（以沱沱河为源），一般称6300公里。流域总面积一百八十余万平方公里，年平均入海水量约九千六百余亿立方米。以干流长度和入海水量论，长江均居世界第三位。",
    "黄河，中国古代也称河，发源于中华人民共和国青海省巴颜喀拉山脉，流经青海、四川、甘肃、宁夏、内蒙古、陕西、山西、河南、山东9个省区，最后于山东省东营垦利县注入渤海。干流河道全长5464千米，仅次于长江，为中国第二长河。黄河还是世界第五长河。",
    "黄河,是中华民族的母亲河。作为中华文明的发祥地,维系炎黄子孙的血脉.是中华民族民族精神与民族情感的象征。",
    "黄河被称为中华文明的母亲河。公元前2000多年华夏族在黄河领域的中原地区形成、繁衍。",
    "在兰州的“黄河第一桥”内蒙古托克托县河口镇以上的黄河河段为黄河上游。",
    "黄河上游根据河道特性的不同，又可分为河源段、峡谷段和冲积平原三部分。 ",
    "黄河,是中华民族的母亲河。"
]
# 设置标签（要给文档打标签的）
doc_labels = ["长江","黄河","黄河","黄河","黄河","黄河","黄河"]

# 语料处理，去标点
sentences = [jieba.lcut(sen) for sen in sentences]
tokenized = []
for sentence in sentences:
    words = []
    for word in sentence:
        if word not in punctuation:
            words.append(word)
    tokenized.append(words)


# 定义数据预处理类，作用是给每个文章添加对应的标签
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):  # enumerate生成坐标和文档
            yield LabeledSentence(words=doc, tags=[self.labels_list[idx]])  # 对每篇文档产生labeledSentence

# 将tokenized的数据进行转换
iter_data = LabeledLineSentence(tokenized, doc_labels)  # 转换后从而得到一个数据集(labeledsentence结构)

# 开始定义模型参数
model = Doc2Vec(dm=1, size=100, window=3, min_count=2, workers=4)  # dm=0/1决定调用DBOW or DM,
# model.save('model')             # 模型保存，因为词向量是构建出的参数们！
# model = Doc2Vec.load('model')   # 模型加载
model.build_vocab(iter_data)      # 从单词频率字典中建立词汇

# 训练，设置迭代1000次，start_alpha为开始学习率，end_alpha与start_alpha线性递减
model.train(iter_data,  total_examples=model.corpus_count, epochs=1000,start_alpha=0.01, end_alpha=0.001)

# 预测
print(model.docvecs.most_similar('黄河'))  # 根据标签找相似的，这里只有黄河和长江，所以结果为长江，并计算出了相似度！是对标签做相似度计算哦！
# 对黄河和长江做相似性计算
print(model.docvecs.similarity('黄河','长江'))  # 计算长江和黄河的相似度！