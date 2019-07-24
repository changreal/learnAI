from gensim.models import Word2Vec
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

# 语料处理，去标点
sentences = [jieba.lcut(sen) for sen in sentences]
tokenized = []  # tokenized是去掉标点的句子
for sentence in sentences:
    words = []
    for word in sentence:
        if word not in punctuation:
            words.append(word)
    tokenized.append(words)

# sg=1 is skip-gram 对低频词敏感;  默认sg=0为CBOW算法那
# size是输出词向量的维数,值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为 100 到 200 之间。
# window是句子中当前词语目标词之间的最大距离,3 表示在目标词前看 3-b 个词，后面看 b 个词（b 在 0-3 之间随机）。
# min_count是对词进行过滤，频率小于 min-count 的单词则会被忽视，默认值为 5。
# negative 和 sample 可根据训练结果进行微调，sample 表示更高频率的词被随机下采样到所设置的阈值，默认值为 1e-3。
# hs=1 表示层级 softmax 将会被使用，默认 hs=0 且 negative 不为 0，则负采样将会被选择使用。
# workers参数控制训练的并行数。
model = Word2Vec(tokenized, sg=1, size=100, window=5, min_count=2, negative=1, sample=0.001, hs=1, workers=4)

# 训练后模型保存与加载，来计算相似性，最大匹配程度！
model.save('model')
model = Word2Vec.load('model') # 加载模型
# model.most_similar(3positive=['女人','女王'], negative=['男人'] ) # 输出[('女王', 0.50882536), ...]
# model.similarity('女人','男人')  # 输出0.73723527
print(model.similarity('黄河', '黄河'))
# # 预测最接近的词，positive与negative的使用（最接近 不接近的词）
print(model.most_similar(positive=['黄河','母亲河'], negative=['长江']))

