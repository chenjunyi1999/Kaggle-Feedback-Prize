import copy

import nltk
from nltk.corpus import wordnet
import random
import numpy as np, os
import pandas as pd, gc
from tqdm import tqdm
nltk.download('wordnet')

random.seed(42)
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
            index = random.randint(0, len(text) - 1)
            n = 0
            while k != label[index]:
                index = random.randint(0, len(text) - 1)
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



# 读取训练集标签
train_df = pd.read_csv('../input/feedback-prize-2021/train.csv')
print(train_df.shape)
train_names, train_texts = [], []
for f in tqdm(list(os.listdir('../input/feedback-prize-2021/train'))):
    train_names.append(f.replace('.txt', ''))
    train_texts.append(open('../input/feedback-prize-2021/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': train_names, 'text': train_texts})
print(train_text_df.head())

num = 1
new_id = []
new_text = []
new_entit = []
all_entities = []

for ii, i in enumerate(train_text_df.iterrows()):
    if ii % 100 == 0:
        print(ii, ', ', end='')
    # i[0]为序号，真正的字典存储在i[1]
    total = i[1]['text'].split().__len__()
    entities = ["O"] * total
    for j in train_df[train_df['id'] == i[1]['id']].iterrows():
        discourse = j[1]['discourse_type']
        list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
        entities[list_ix[0]] = f"B-{discourse}"
        for k in list_ix[1:]: entities[k] = f"I-{discourse}"
    all_entities.append(entities)
    if 'I-Counterclaim' in entities or 'I-Rebuttal' in entities:
        for jk in range(2):
            text = copy.deepcopy(i[1]['text'])
            entit = copy.deepcopy(entities)
            radint = random.randrange(0, 10)
            if radint < 5:
                text = word_level_aug(text, 0.3)
            else:
                text, entit = ri_rs_rd(text, entit, 0.2)
            new_id.append(num)
            new_text.append(text)
            new_entit.append(entit)
            num += 1

train_text_df['entities'] = all_entities
print(len(train_text_df))
for i in range(len(new_id)):
    train_text_df.loc[len(train_text_df)] = [new_id[i], new_text[i], new_entit[i]]
print(len(train_text_df))
train_text_df.to_csv('../input/train_NER_augment_counter_and_rebuttal.csv', index=False)


# todo 句子级别同义、增删、翻转

