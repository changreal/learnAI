import pickle
import json

"""
HMM需要的是 
1. 状态值集合，比如下面的{'B','M','E','S'} B首字母  M中间字 E结束字 S单个字
2. 观察值集合(所有语料的汉字)
3. 转移概率矩阵  就是从状态 X 转移到状态Y 的概率，是一个{}*{}的矩阵
4. 发射概率矩阵  每个元素都是一个条件概率！ P(观测值i | 状态值j)
5. 初始状态分布   比如表示句子的第一个字属于{BMES}这四种状态的概率
HMM应用在分词上，2-5是参数，求的是状态值序列；最有名的是Viterbi算法
这是预测问题，已知λ=(A,B,π)，已知T，已知O，求状态
"""

"""
预定义-  状态、eps、停顿标点符号
"""
STATES = {'B','M','E','S'}   # 初始装填
EPS = 0.0001                 # EPS是不在时，返回的默认值
seg_stop_words = {" ","，","。","“","”",'“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’", "──", ",", ".", "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", "[", "]", "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n","\t"}


"""
封装成类
- 包括初始化、保存、加载、训练、计算、预测
"""
class HMM_Model:
    def __init__(self):
        # 这些都是dict，dict可以遍历，并且要记录字符串的value 所以用dict！
        self.trans_mat = {}  # 状态转移矩阵，trans_mat[state1][state2] 表示训练集中由 state1 转移到 state2 的次数。
        self.emit_mat   = {}  # 观测矩阵，emit_mat[state][char] 表示训练集中单字 char 被标注为 state 的次数。
        self.init_vec   = {}  # 初始状态分布向量，init_vec[state] 表示状态 state 在训练集中出现的次数。
        self.state_count = {} # 状态统计向量，state_count[state]表示状态 state 出现的次数。
        self.states     = {}  # 状态值集合
        self.word_set   = {}  # 词集合，包括所有单词
        self.inited     = False  # 初始化标志

    # 初始化数据结构
    def setup(self):
        for state in self.states:      # 遍历状态
            # 初始化状态发射矩阵
            self.emit_mat[state] = {}  # {{},{},..{}}
            # 初始化状态分布向量
            self.init_vec[state] = 0   # {0,0,...0}
            # 初始化状态统计向量
            self.state_count[state] = 0
            # 初始化状态转移矩阵
            self.trans_mat[state] = {}  # {{}.{},...{}}
            for target in self.states:  # 遍历目标状态
                self.trans_mat[state][target] = 0.0
        self.inited = True

    # 模型保存
    def save(self, filename='hmm.json', code='json'):
        """
        保存训练好的模型
        :param filename: 默认名词为hmm.json
        :param code: 默认两种保存格式 pickle or json
        :return:
        """
        fw = open(filename, 'w', encoding='utf-8')
        data = {
            "trans_mat" : self.trans_mat,
            "emit_mat"  : self.emit_mat,
            "init_vec"  : self.init_vec,
            "state_count" : self.state_count
        }
        if code == 'json':
            txt = json.dumps(data)
            txt = txt.encode('utf-8').decode('unicode-escape')  # 转为unicode格式
            fw.write(txt)
        elif code == 'pickle':
            pickle.dump(data, fw)
        fw.close()

    # 模型加载
    def load(self, filename, code):
        fr = open(filename, 'r', encoding='utf-8')
        if code == 'json':
            txt = fr.read()
            model = json.loads(txt)
        elif code == 'pickle':
            model = pickle.load(fr)
        self.trans_mat = model["trans_mat"]
        self.emit_mat = model["emit_mat"]
        self.init_vec = model["init_vec"]
        self.state_count = model["state_count"]
        self.inited = True
        fr.close()

    # 模型训练（依旧以后的分词训练集进行的训练）
    def do_train(self, observes, states):
        """
        输入观测序列and状态序列进行训练，依次更新矩阵数据，参数是频数不是频率
        这样的设计使得模型可以进行在线训练，随时能接受新训练数据继续训练，不会丢失前次训练结果
        :param observes: 训练集的观测序列    list ['的’,'骚'....]
        :param states:   训练集的状态序列    list ['S','E','S','M','E']等
        :return:
        """
        if not self.inited:
            self.setup()

        for i in range(len(states)):
            if i==0:  # 第一个状态
                self.init_vec[states[0]] += 1    # 计数+1
                self.state_count[states[0]] += 1 # 计数+1
            else:
                self.trans_mat[states[i-1]][states[i]]  += 1  # 上个状态向下个状态转移的频数
                self.state_count[states[i]] += 1  # 状态计数
                if observes[i] not in self.emit_mat[states[i]]:
                    self.emit_mat[states[i]][observes[i]] = 1
                else:
                    self.emit_mat[states[i]][observes[i]] += 1

    # 频数转频率
    def get_prob(self):
        """
        在预测前，将结局结构的频数转换为频率
        :return:
        """
        init_vec = {}  # <class 'dict'>: {'E': 0, 'B': 788187, 'M': 0, 'S': 282095}
        trans_mat = {}
        emit_mat = {}
        default = max(self.state_count.values())  # 找出出现频率最高的状态

        # 这里将init_vec的值转换为频率
        for key in self.init_vec:  # {'E': 0, 'B': 788187, 'M': 0, 'S': 282095}
            if self.state_count[key] != 0:  # init_vec只有S B两种状态
                init_vec[key] = float(self.init_vec[key]) / self.state_count[key]
            else:
                init_vec[key] = float(self.init_vec[key]) / default

        for key1 in self.trans_mat:
            trans_mat[key1] = {}  # 也是字典形式
            for key2 in self.trans_mat[key1]:
                if self.state_count[key1] != 0:
                    trans_mat[key1][key2] = float(self.trans_mat[key1][key2]) / self.state_count[key1]
                else:
                    trans_mat[key1][key2] = float(self.trans_mat[key1][key2]) / default

        for key1 in self.emit_mat:
            emit_mat[key1] = {}
            for key2 in self.emit_mat[key1]:
                if self.state_count[key1] != 0:
                    emit_mat[key1][key2] = float(self.emit_mat[key1][key2]) / self.state_count[key1]
                else:
                    emit_mat[key1][key2] = float(self.emit_mat[key1][key2]) / default

        return init_vec, trans_mat, emit_mat

    # 模型预测，采用viterbi算法求最优路径！
    def do_predict(self, sequence):  # 比如sequence='中国的人工智能发展进入高潮阶段！'
        tab = [{}]
        path = {}
        init_vec, trans_mat, emit_mat = self.get_prob()  # 得到概率形式的三要素

        # 初始化 路径初始所需值
        for state in self.states:  # <class 'set'>: {'E', 'B', 'M', 'S'}
            tab[0][state] = init_vec[state] * emit_mat[state].get(sequence[0], EPS)  # 获取第一个值，比如P(E)*P(中，0.001)，得到一开始的通路
            path[state] = [state]  # 已知通路

        # 创建动态搜索表
        for t in range(1, len(sequence)):
            tab.append({})
            new_path = {}
            for state1 in self.states:
                items = []
                for state2 in self.states:
                    if tab[t - 1][state2] == 0:
                        continue
                    prob = tab[t - 1][state2] * trans_mat[state2].get(state1, EPS) * emit_mat[state1].get(sequence[t], EPS)  # 这个不是很懂捏
                    items.append((prob, state2))
                best = max(items)
                tab[t][state1] = best[0]  # 找出最佳
                new_path[state1] = path[best[1]] + [state1]  # 更新新的路径
            path = new_path

        # 搜索最优路径
        prob, state = max([(tab[len(sequence) - 1][state], state) for state in self.states])
        return path[state]


"""
词标注
"""
# 定义工具函数, 用在训练数据的标注
def get_tags(src):
    tags = []
    if len(src) == 1:
        tags = ['S']
    elif len(src) == 2:
        tags = ['B','E']
    else:
        m_num = len(src)-2  # 中间部分
        tags.append('B')
        tags.extend(['M']*m_num)
        tags.append('E')
    return tags

# 将输入的句子分割为词语列表（也就是预测得到的状态序列）解析成list列表返回
def cut_sent(src, tags):
    """
    传入句子 与 分词标注的状态，把句子分词了
    :param src: 中国的人工智能发展进入高潮阶段！'
    :param tags: list'>: ['B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B']
    :return:
    """
    word_list = []
    start = -1
    started = False  # 标记是是否开始了

    if len(tags) != len(src):   # tags长度和src长度要相同！
        return None

    if tags[-1] not in {'S', 'E'}:
        if tags[-2] in {'S', 'E'}: # 倒2的标注是 S 或者 E
            tags[-1] = 'S'  # 修正最后一个成S的情况。也就是比如最后一个是符号！，把！标注为B，这不对 修正为S
        else:
            tags[-1] = 'E'

    for i in range(len(tags)):
        if tags[i] == 'S':
            if started:
                started = False
                word_list.append(src[start:i])
            word_list.append(src[i])
        elif tags[i] == 'B':
            if started:
                word_list.append(src[start:i])
            start = i  # 标记词开始的位置
            started = True
        elif tags[i] == 'E':
            started = False
            word = src[start:i + 1]  # 词为当前位置开始+1，因为最后一个取不到)
            word_list.append(word)
        elif tags[i] == 'M':
            continue
    return word_list


"""
定义分词器类HMMsoyoger, 继承HMM_Model类实现中文分词训练、分词功能
"""
class HMMSoyoger(HMM_Model):
    def __init__(self, *args, **kwargs):
        super(HMMSoyoger, self).__init__(*args, **kwargs)
        self.states = STATES
        self.data = None

    # 加载训练数据
    def read_text(self, filename):
        self.data = open(filename, 'r', encoding="utf-8")

    # 模型训练
    def train(self):
        """
        根据单词生成观测序列 and 状态序列，通过父类do_train方法训练
        :return:
        """
        if not self.inited:
            self.setup()
        for line in self.data:
            line = line.strip()
            if not line:
                continue

            # 观察序列
            observes = []
            for i in range(len(line)):    # 读入词
                if line[i] == " ":        # 原数据中，分词就是以空格分开的了！
                    continue
                observes.append(line[i])  # 生成<class 'list'>: ['4', '日', '清', '晨', '，', '新', '县']形式

            # 状态序列
            words  = line.split(" ")      # word是分词的形式 原来以空格分开的分词了
            states = []
            for word in words:
                if word in seg_stop_words:
                    continue
                states.extend(get_tags(word))  # 仅仅标注当前当前词的结构，(分词结束or中间or开始) 返回每个字的list，<class 'list'>: ['S', 'S', 'B', 'E', 'B', 'E', 'S']

             # 开始训练
            if(len(observes) >= len(states)):  # 理论上observes长度要大的，因为states里过滤了停用词 比如','
                self.do_train(observes, states)
            else:
                pass

    # 模型分词预测
    def lcut(self, sentence):
        # 模型训练好之后，进行分词测试
        try:
            tags = self.do_predict(sentence)  # 形如：<class 'list'>: ['B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B']
            return cut_sent(sentence, tags)  # 传入句子与分词 测试
        except:
            return sentence

# 训练模型
soyoger = HMMSoyoger() # 实例化HMM分词器类
soyoger.read_text("./syj_trainCorpus_utf8/syj_trainCorpus_utf8.txt")
soyoger.train()
# 测试
wordlist = soyoger.lcut("中国的人工智能发展进入高潮阶段！")
print(wordlist)
wordlist2 = soyoger.lcut("中文自然语言处理是人工智能技术的一个重要分支。")
