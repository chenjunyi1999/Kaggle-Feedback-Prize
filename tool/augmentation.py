import nltk
from nltk.corpus import wordnet
import random
import numpy as np
#nltk.download('wordnet')

# 有待测试
def word_level_aug(text, ratio=0.5):
    text = text.split()
    length = len(text)
    num_changes = int(length*ratio)
    # 生成可能含有相同的
    # idxs = np.random.randint(0, length-1, size=num_changes)
    # 生成肯定不相同的
    idxs = random.sample(range(0, length-1), num_changes)
    for idx in idxs:
        if text[idx][-1] in ['\'','"','.',',','?','!','......','...']:
            continue
        idx_synonyms = []
        for syn in wordnet.synsets(text[idx]):
            for lm in syn.lemmas():
                idx_synonyms.append(lm.name())
        if len(idx_synonyms)<1:
            continue
        text[idx] = random.sample(idx_synonyms, 1)[0]
    return ' '.join([i for i in text])

def randon_insection(text, label, ratio):
    text = text.split()
    length = len(text)
    num_changes = [int(length * rat) for rat in ratio]
    # 随机插入
    for i in range(num_changes[0]):
        idx = random.randint(0, len(text) - 1)
        k = label[idx]
        if text[idx][-1] in ['\'', '"', '.', ',', '?', '!', '......', '...']:
            continue
        idx_synonyms = []
        for syn in wordnet.synsets(text[idx]):
            for lm in syn.lemmas():
                idx_synonyms.append(lm.name())
        if len(idx_synonyms) < 1:
            continue
        index = random.randint(0, len(text))
        text.insert(index, random.sample(idx_synonyms, 1)[0])
        label.insert(index, label[idx])

    # 随机交换
    for i in range(num_changes[1]):
        idx1 = random.randint(0, len(text) - 1)
        idx2 = random.randint(0, len(text) - 1)
        text[idx1], text[idx2] = text[idx2], text[idx1]
        label[idx1], label[idx2] = label[idx2], label[idx1]

    # 随机删除
    for i in range(num_changes[2]):
        idx = random.randint(0, len(text) - 1)
        text.pop(idx)
        label.pop(idx)

    return ' '.join([i for i in text]), label

print(randon_insection('one person says to do.', [1,2,3,4,5], [0.15, 0.5,0.15]))
# todo 句子级别同义、增删、翻转

