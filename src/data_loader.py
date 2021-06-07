import os

import torch

# 本文件主要是给训练、测试过程提供便捷的取样本操作


categories = ['财经', '房产', '教育', '科技', '军事', '汽车', '体育', '游戏', '娱乐']
# 类别名对应的id
labels = {'财经': 0, '房产': 1, '教育': 2, '科技': 3, '军事': 4, '汽车': 5,
          '体育': 6, '游戏': 7, '娱乐': 8}

#这两个改成sample数量-1
MAX_TRAIN_INDEX = 136
MAX_TEST_INDEX = 70

# index 0~235
# labels = [..] n个标签, words = [[]..] n个[],[]内50个词索引
module_path = os.path.dirname(__file__)


# 在./data/XXXsamples文件夹下，一个sample= [(),()....] 约100个()
# 一个 () = (类别id,[word1_index, word2_index...]), [word1_index..]为新闻分词后，每个词在word2vec模型中对应的索引值
# 类别id见labels中定义的变量

# trainSamples 是训练数据
# testSamples 测试，从未训练过

# 100个新闻样本, tuple(list,list)
# 第一个list 是100个样本对应的类别id, 后面的list则保存了每条新闻分词后每个词的索引,即[[新闻1的词索引..][新闻2..]]
# return ([label_1,label_2,..,label_100], [[news_1_word_1_index,..,news_1_word_50_index]...[news_100_word_1_index,..] )
def get_lw_of_train_sample(index):
    file = module_path + '/data/trainSamples/sample-' + str(index) + '.pth'
    return get_lw_sample(file)


def get_lw_of_test_sample(index):
    file = module_path + '/data/testSamples/sample-' + str(index) + '.pth'
    return get_lw_sample(file)


def get_lw_sample(file):
    sample = torch.load(file)
    labels = [l for l, w in sample]  # 提取label和words
    words = [w for l, w in sample]
    return labels, words


if __name__ == '__main__':
    # 可以看看具体的数据
    for i in range(0, MAX_TEST_INDEX + 1):
        l, w = get_lw_of_test_sample(i)
        print(l)
