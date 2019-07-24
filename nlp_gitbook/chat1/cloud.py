#引入所需要的包
import jieba
import pandas as pd 
import numpy as np
from scipy.misc import imread 
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt

# 文件引入
dir = './wordcloud/'
file = "".join([dir, 'z_m.csv'])
stop_words = "".join([dir,"stopwords.txt"])
simhei = "".join([dir,"simhei.ttf"])              # 字体


# 读取语料与处理
df = pd.read_csv(file, encoding='utf-8')
df.head()
df.dropna(inplace=True)
# content2 = df.content.values
content = df.content.values.tolist()   # .content.values是ndarray，转换为list
# print(df.content.values.tolist())

# 分词
segment=[]
for line in content:
    try:
        segs=jieba.cut_for_search(line)  # 搜索模式
        segs = [v for v in segs if not str(v).isdigit()]#去数字
        segs = list(filter(lambda x:x.strip(), segs))   #去左右空格
        #segs = list(filter(lambda x:len(x)>1, segs)) #长度为1的字符
        for seg in segs:
            if len(seg)>1 and seg!='\r\n':
                segment.append(seg)
    except:
        print(line)
        continue
# print(segment)

# 加入DataFrame
words_df = pd.DataFrame({'segment': segment})
stopwords=pd.read_csv(stop_words,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')                 # 加入停用词
#安装关键字groupby分组统计词频，并按照计数降序排序
words_stat=words_df.groupby(by=['segment'])['segment'].agg({"计数":np.size})
words_stat=words_stat.reset_index().sort_values(by=["计数"],ascending=False)
#分组之后去掉停用词
words_stat=words_stat[~words_stat.segment.isin(stopwords.stopword)]

#下面是重点，绘制wordcloud词云，这一提供2种方式
#第一种是默认的样式
wordcloud=WordCloud(font_path=simhei,background_color="white",max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
wordcloud.to_file('wordcloud_1.jpg')  #保存结果

#第二种是自定义图片
text = " ".join(words_stat['segment'].head(100).astype(str))  # 前100个词用空格连接生成一个字符串
abel_mask = imread(''.join([dir,"china.jpg"]))  #这里设置了一张中国地图
wordcloud2 = WordCloud(background_color='white',  # 设置背景颜色
                     mask = abel_mask,  # 设置背景图片
                     max_words = 3000,  # 设置最大现实的字数
                     font_path = simhei,  # 设置字体格式
                     width=2048,
                     height=1024,
                     scale=4.0,
                     max_font_size= 300,  # 字体最大值
                     random_state=42).generate(text)

# 根据图片生成词云颜色
image_colors = ImageColorGenerator(abel_mask)
wordcloud2.recolor(color_func=image_colors)
# 以下代码显示图片
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
wordcloud2.to_file(r'wordcloud_2.jpg') #保存结果