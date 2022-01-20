import os

# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
from torch.utils.data import DataLoader, Dataset
from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0,1,2,3 for four gpu

# VERSION FOR SAVING MODEL WEIGHTS
VER = 3
# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = '../input'

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = None
# LOAD_MODEL_FROM = './'

# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = '../model/bigbird'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = '../model/bigbird'

#################################################################
# config
from torch import cuda

config = {'model_name': MODEL_NAME,
          'max_length': 1024,
          'train_batch_size': 6,
          'valid_batch_size': 6,
          'epochs': 5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm': 10,
          'device': 'cuda' if cuda.is_available() else 'cpu'}

# THIS WILL COMPUTE VAL SCORE DURING COMMIT BUT NOT DURING SUBMIT
COMPUTE_VAL_SCORE = True
if len(os.listdir('../input/feedback-prize-2021/test')) > 5:
    COMPUTE_VAL_SCORE = False

################################################################
# 加载数据
import numpy as np, os
import pandas as pd, gc
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, AutoModel
import torch
from sklearn.metrics import accuracy_score

# 读取训练集标签
train_df = pd.read_csv('../input/feedback-prize-2021/train.csv')
print(train_df.shape)
# 输出前5条数据
print(train_df.head())

# 读取测试集文本
test_names, test_texts = [], []
for f in list(os.listdir('../input/feedback-prize-2021/test')):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open('../input/feedback-prize-2021/test/' + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})
print(test_texts.head())

# 读取训练集文本
train_names, train_texts = [], []
for f in tqdm(list(os.listdir('../input/feedback-prize-2021/train'))):
    train_names.append(f.replace('.txt', ''))
    train_texts.append(open('../input/feedback-prize-2021/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': train_names, 'text': train_texts})
print(train_text_df.head())

###############################################################
# 转化为ner标签
# LOAD_TOKENS_FROM 为存储ner标签csv的路径，处理一次，以后就不用处理了
# Ner标签采用BIO标记法
if not LOAD_TOKENS_FROM:
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
    train_text_df['entities'] = all_entities
    train_text_df.to_csv('../input/train_NER.csv', index=False)

else:
    from ast import literal_eval

    train_text_df = pd.read_csv(f'{LOAD_TOKENS_FROM}/train_NER.csv')
    # pandas saves lists as string, we must convert back
    # 将字符串形的list转化会list
    train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x))

print(train_text_df.shape)
train_text_df.head()

# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = [
    'O',
    'B-Lead', 'I-Lead',
    'B-Position', 'I-Position',
    'B-Claim', 'I-Claim',
    'B-Counterclaim', 'I-Counterclaim',
    'B-Rebuttal', 'I-Rebuttal',
    'B-Evidence', 'I-Evidence',
    'B-Concluding Statement', 'I-Concluding Statement']
# 存储标签与index之间的映射
labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}

#########################################################################
# 定义dataset
import nltk
from nltk.corpus import wordnet
import random

LABEL_ALL_SUBTOKENS = True

def correct(words, type=0):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    # 是字符，也返回字符
    if type == 0:
        words = words.split(' ')
        for i in range(len(words)):
            cur_word = spell.correction(words[i])
            if words[i] != cur_word:
                if words[i][-1] in commas:
                    continue
                else:
                    words[i] = cur_word
        return ' '.join(words)

    # 是列表，也返回列表
    elif type == 1:
        for i in range(len(words)):
            cur_word = spell.correction(words[i])
            if words[i] != cur_word:
                words[i] = cur_word
        return words
    else:
        print("type error")
        return

def word_level_aug(text, ratio=0.15):
    text = text.split()
    length = len(text)
    num_changes = int(length*ratio)
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


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, get_wids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids  # for validation，是否创建target

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS
        text = self.data.text[index]
        word_labels = self.data.entities[index] if not self.get_wids else None
        if not self.get_wids:
            random.seed(42)
            randint = random.randrange(0,2)
            if randint == 1:
                radint = random.randrange(0, 10)
                if radint < 4:
                    text = word_level_aug(text, 0.5)
                else:
                    text, word_labels = ri_rs_rd(text, word_labels, 0.15)


        # TOKENIZE TEXT
        # is_split_into_words:假设输入已经按字切分，直接进行tokenize，适用于ner
        encoding = self.tokenizer(text.split(),
                                  is_split_into_words=True,
                                  # return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)
        word_ids = encoding.word_ids()
        # CREATE TARGETS
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels_to_ids[word_labels[word_idx]])
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append(labels_to_ids[word_labels[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            encoding['labels'] = label_ids

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids:
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)

        return item

    def __len__(self):
        return self.len


#################################################################
# 创建训练集和验证集：9：1
# CHOOSE VALIDATION INDEXES (that match my TF notebook)
IDS = train_df.id.unique()
print('There are', len(IDS), 'train texts. We will split 90% 10% for validation.')

# TRAIN VALID SPLIT 90% 10%
np.random.seed(42)
train_idx = np.random.choice(np.arange(len(IDS)), int(0.9 * len(IDS)), replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)), train_idx)
np.random.seed(None)

# CREATE TRAIN SUBSET AND VALID SUBSET
data = train_text_df[['id', 'text', 'entities']]
train_dataset = data.loc[data['id'].isin(IDS[train_idx]), ['text', 'entities']].reset_index(drop=True)
test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH)
training_set = dataset(train_dataset, tokenizer, config['max_length'], False)
testing_set = dataset(test_dataset, tokenizer, config['max_length'], True)

# TRAIN DATASET AND VALID DATASET
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 2,
                'pin_memory': True
                }

test_params = {'batch_size': config['valid_batch_size'],
               'shuffle': False,
               'num_workers': 2,
               'pin_memory': True
               }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# TEST DATASET
test_texts_set = dataset(test_texts, tokenizer, config['max_length'], True)
test_texts_loader = DataLoader(test_texts_set, **test_params)


######################################################################
# 模型训练
# 训练五个epoch，每个epoch的学习率都一次下降

# CREATE MODEL
class MyModel(nn.Module):
    def __init__(self, freeze_bert=False, model_name='bert-base-chinese', hidden_size=768, num_classes=2):
        super(MyModel, self).__init__()
        # output_hidden_states=True输出每一层transformer的输出，但是只有最后一层为常用word embedding
        self.automodel = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, num_classes, bias=False),
        )

        if freeze_bert:
            for p in self.automodel.parameters():
                p.requires_grad = False
            for p in self.fc.parameters():
                p.requires_grad = False
        '''
        self.is_con_re = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 3, bias=False),
        ) # 1是'Counterclaim', 2是'Rebuttal'
        '''

    def forward(self, input_ids, attn_masks):
        outputs = self.automodel(input_ids, attention_mask=attn_masks)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]),
                                  dim=-1)  # [bs, seq_len, hidden_dim*4]
        #first_hidden_states = hidden_states[:, :, :]  # [bs, hidden_dim*4]
        logits = self.fc(hidden_states)
        logits = torch.softmax(logits, dim=-1)
        #log = self.is_con_re(hidden_states)
        #log = torch.softmax(log, dim=-1)
        return logits # , log


model = MyModel(model_name=DOWNLOADED_MODEL_PATH, num_classes=15, freeze_bert=False)
optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rates'][0])
model.to(config['device'])

def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []

    # put model in training mode
    model.train()
    lossn = nn.CrossEntropyLoss()
    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.long)

        tr_logits = model(input_ids=ids, attn_masks=mask)
        #tr_logits, is_two = model(input_ids=ids, attn_masks=mask)
        loss = 0
        labels2 = []
        '''
        for i1 in range(len(labels)):
            label = []
            for i2 in range(len(labels[i1])):
                if labels[i1][i2] == -100:
                    label.append(-100)
                    continue
                if 'Counterclaim' in ids_to_labels[int(labels[i1][i2])]:
                    label.append(1)
                elif 'Rebuttal' in ids_to_labels[int(labels[i1][i2])]:
                    label.append(2)
                else:
                    label.append(0)
            labels2.append(label)
        '''
        for i in range(len(tr_logits)):
            loss += lossn(tr_logits[i], labels[i])
            #loss += lossn(is_two[i], torch.tensor(labels2[i]).to(config['device']))

        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 200 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss after {idx:04d} training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, 15)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        # tr_labels.extend(labels)
        # tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


# 加载\训练模型
if not LOAD_MODEL_FROM:
    # 加载模型部分参数
    '''
    path = './bigbird_v1.pt'
    save_model = torch.load(path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    '''
    for epoch in range(config['epochs']):

        print(f"### Training epoch: {epoch + 1}")
        for g in optimizer.param_groups:
            g['lr'] = config['learning_rates'][epoch]
        lr = optimizer.param_groups[0]['lr']
        print(f'### LR = {lr}\n')

        train(epoch)
        torch.cuda.empty_cache()
        # gc释放内存
        gc.collect()

    torch.save(model.state_dict(), f'bigbird_v{VER}.pt')
else:
    model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/bigbird_v{VER}.pt'))
    print('Model loaded.')


#################################################################################
# 推理及验证数据
proba_thresh = {
    "Lead": 0.7,
    "Position": 0.55,
    "Evidence": 0.65,
    "Claim": 0.55,
    "Concluding Statement": 0.7,
    "Counterclaim": 0.5,
    "Rebuttal": 0.55,
}

min_thresh = {
    "Lead": 9,
    "Position": 5,
    "Evidence": 14,
    "Claim": 3,
    "Concluding Statement": 11,
    "Counterclaim": 6,
    "Rebuttal": 4,
}

def inference(batch):
    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])
    #outputs, is_two = model(ids, attn_masks=mask)
    outputs = model(ids, attn_masks=mask)

    all_preds = torch.argmax(outputs, axis=-1).cpu().numpy()
    #twos = torch.argmax(is_two, axis=-1).cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    sorces = []
    for k, text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        sorce = []
        word_ids = batch['wids'][k].numpy()
        previous_word_idx = -1
        for idx, word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                prediction.append(token_preds[idx])
                sorce.append(outputs[k][idx][text_preds[idx]])
                previous_word_idx = word_idx
        predictions.append(prediction)
        sorces.append(sorce)

    return predictions, sorces
    #return predictions, sorces, twos


# 在推理阶段，对每个字词都进行预测。所以在最后得到输出标签时，要将这些自词合并
# TODO 尝试不同的子词标签合并方法

def get_predictions(df=test_dataset, loader=testing_loader):
    # put model in training mode
    model.eval()

    # GET WORD LABEL PREDICTIONS
    y_pred2 = []
    y_log = []
    # y_two = []
    for batch in loader:
        #labels, logs, twos = inference(batch)
        labels, logs = inference(batch)
        y_pred2.extend(labels)
        y_log.extend(logs)
        # y_two.extend(twos)

    final_preds2 = []
    for i in range(len(df)):

        idx = df.id.values[i]
        # pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
        pred = y_pred2[i]  # Leave "B" and "I"
        grade = y_log[i]
        # twoes = y_two[i]
        preds = []
        j = 0
        torch.set_printoptions(threshold=np.inf)
        '''
        print([twoes[i] for i in range(len(twoes)) if twoes[i]!=0])
        '''
        while j < len(pred):
            cls = pred[j]
            # is_x = twoes[j]
            sum_prd = 0

            '''
            if is_x == 0:
                pass
            else:
                start = j
                stop = j + 1
                while stop < len(twoes) and twoes[stop] == is_x:
                    stop += 1
                if stop - start >= 10:
                    if is_x == 2:
                        final_preds2.append((idx, 'Rebuttal',
                                         ' '.join(map(str, list(range(start, stop))))))
                    else:
                        final_preds2.append((idx, 'Counterclaim',
                                             ' '.join(map(str, list(range(start, stop))))))
                    j = stop
                    continue
            '''
            if cls == 'O':
                j += 1
            else:
                cls = cls.replace('B', 'I')  # spans start with B
                sum_prd += grade[j]
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                sum_prd += grade[end]
                end += 1

            if cls != 'O' and cls != '' and sum_prd/(end - j) >= proba_thresh[cls.replace('I-', '')] and end - j >= min_thresh[cls.replace('I-', '')]:
                final_preds2.append((idx, cls.replace('I-', ''),
                                     ' '.join(map(str, list(range(j, end))))))

            j = end

    oof = pd.DataFrame(final_preds2)
    oof.columns = ['id', 'class', 'predictionstring']

    return oof


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id', 'class'],
                           right_on=['id', 'discourse_type'],
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score, TP, FP, FN


if COMPUTE_VAL_SCORE:
    # note this doesn't run during submit
    # VALID TARGETS
    valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

    # OOF PREDICTIONS
    oof = get_predictions(test_dataset, testing_loader)

    # COMPUTE F1 SCORE
    f1s = []
    CLASSES = oof['class'].unique()
    print()
    for c in CLASSES:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = valid.loc[valid['discourse_type'] == c].copy()
        f1, TP, FP, FN = score_feedback_comp(pred_df, gt_df)
        print(c, f1, "TP: ", TP, ", FP: ", FP, ' FN: ', FN)
        f1s.append(f1)
    print()
    print('Overall', np.mean(f1s))
    print()

# 生成测试结果
sub = get_predictions(test_texts, test_texts_loader)
sub.head()
sub.to_csv("submission.csv", index=False)
