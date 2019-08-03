"""
模型参数配置————预先定义模型参数和加载语料以及模型保存名称
"""
class Config(object):
    poetry_file = 'poetry.txt'       # 语料文本，处理后的语料文本，只用五言绝句！
    weight_file = 'poetry_model.h5'  # 保存模型的文件名
    # 根据前6个字预测第7个字
    max_len = 6             # 跨度
    batch_size = 32         # 批量
    learning_rate = 0.001
