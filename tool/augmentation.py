import nltk
from nltk import wordnet
import random
import numpy as np
nltk.download('wordnet')

# 有待测试
def word_level_aug(text, ratio=0.15):
    text = text.split()
    length = len(text)
    num_changes = int(length*ratio)
    # 生成可能含有相同的
    idxs = np.random.randint(0, length-1, size=num_changes)
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
        text[idx] = random.sample(idx_synonyms, 1)
    return ''.join(text)

# todo 句子级别同义、增删、翻转

