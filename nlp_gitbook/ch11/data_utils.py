"""
预处理
文字转换为one-hot形式
"""
puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']
def preprocess_file(Config):
    # 语料文本内容
    files_content = ''
    with open(Config.poetry_file, 'r', encoding='utf-8') as f:
        # 注释掉以下有bug的部分
        # for line in f:
        #     for char in puncs:
        #         line = line.replace(char, "")   # 每行的文本有不可用的替代字符都要去掉，也就是只有，和。
        #     files_content += line.strip() + ']' # '芝田初雁去，绮树巧莺来。]何必汾阳处，始复有山泉。]' 这种形式
        #
        for line in f:
            x = line.strip() + ']'   # 原来的没有去掉空格。and直接在结尾+']'就可以表示诗结束
            x = x.split(':')[1]      # 原来诗歌的格式是，标题:诗歌内容] 这三个~
            if len(x) < 5:           # 删除掉非五言的
                continue
            if x[5] == '，':         # 筛选有效句子，去掉这种情况：“唐享昊天乐。第七:尊浮九酝，礼备三周。陈诚菲奠，契福神猷。”
                files_content += x

    # words是list词典，里面会重复（只是为了制作成词频）
    words = sorted(list(files_content))  # 开始制作词典哈,但是会重复，然后会把标点符号和数字排序到前面a
    # words.remove(']')  # 这句也注释掉了
    # 词频字典
    counted_words = {} # dict

    # 计算词频
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉字典里的低频词
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]  # 把list里的低频词都删掉
    # del counted_words[']']  # 不计算]的词频

    # 得到排序后的词频字典
    wordPairs = sorted(counted_words.items(), key=lambda x:-x[  1])  #  字典按词频排序
    words,_  = zip(*wordPairs)  # 得到所需要的词

    # word到id的映射
    # word2num = dict((c, i+1) for i, c in enumerate(words))
    word2num = dict((c,i) for i, c in enumerate(words))  # 改为i了哈，这里没+1，那可能没见过的词用其他位置来表达
    num2word = dict((i,c) for i,c in enumerate(words))  # {0: '好', 1: '对', 2: '我', 3: '破'}
    # word2numF = lambda x: word2num.get(x,0)  # 获取指定词的num值（词表示）
    word2numF = lambda x: word2num.get(x,len(words)-1)  # dict的get方法函数返回指定键的值，如果不在字典中则返回默认值，这里是如果词不在字典中，返回词典的最后一个词

    return word2numF, num2word, words, files_content
