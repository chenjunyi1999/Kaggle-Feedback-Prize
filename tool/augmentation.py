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

def ri_rs_rd(text, label, ratio):
    text = text.split()
    length = len(text)
    num_changes = int(length * ratio)
    k = random.randint(0, 2)
    if k == 0:
        # 随机插入
        for i in range(num_changes):
            idx = random.randint(0, len(text) - 1)
            k = label[idx]
            n = 0
            while 'B-' in k:
                idx = random.randint(0, len(text) - 1)
                k = label[idx]
                n += 1
                if n > 10:
                    break
            n = 0
            while text[idx][-1] in ['\'','"','.',',','?','!','......','...']:
                idx = random.randint(0, len(text) - 1)
                k = label[idx]
                if n > 10:
                    break
            idx_synonyms = []
            for syn in wordnet.synsets(text[idx]):
                for lm in syn.lemmas():
                    idx_synonyms.append(lm.name())
            if len(idx_synonyms)<1:
                continue
            index = random.randint(0, len(text))
            n = 0
            while k != label[index]:
                index = random.randint(0, len(text))
                if n > 10:
                    break
            text.insert(index, random.sample(idx_synonyms, 1)[0])
            label.insert(index, label[idx])

    if k == 1:
        # 随机交换
        for i in range(num_changes):
            idx1 = random.randint(0, len(text) - 1)
            idx2 = random.randint(0, len(text) - 1)
            n = 0
            while label[idx1] != label[idx2]:
                idx2 = random.randint(0, len(text) - 1)
                if n > 10:
                    break
            text[idx1], text[idx2] = text[idx2], text[idx1]
            label[idx1], label[idx2] = label[idx2], label[idx1]

    if k == 2:
        #随机删除
        for i in range(num_changes):
            idx = random.randint(0, len(text) - 1)
            n = 0
            while 'B-' in label[idx]:
                idx = random.randint(0, len(text) - 1)
                if n > 10:
                    break
            text.pop(idx)
            label.pop(idx)

    return ' '.join([i for i in text]), label

print(ri_rs_rd('one person says to do.', ['B-a', 'I-a','I-a','B-2','I-2'], 0.5))
# todo 句子级别同义、增删、翻转

