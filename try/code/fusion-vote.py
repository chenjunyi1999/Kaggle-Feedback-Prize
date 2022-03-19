import copy
import os

# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
from torch.utils.data import DataLoader, Dataset
from torch import nn
import nltk
from nltk.corpus import wordnet
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0,1,2,3 for four gpu

# VERSION FOR SAVING MODEL WEIGHTS
VER = 8
# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = '../input'

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL

LOAD_MODEL_FROM = './'

# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = '../model/longformer/model'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = '../model/longformer/model'

#################################################################
# config
from torch import cuda

config = {'model_name': MODEL_NAME,
          'max_length': 2048,
          'train_batch_size': 2,
          'valid_batch_size': 4,
          'epochs': 3,
          'learning_rates': [2.5e-6, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-7, 2.5e-7, 2.5e-7, 2.5e-7],
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

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, AutoModel, LongformerTokenizerFast
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
    # train_text_df = pd.read_csv(f'{LOAD_TOKENS_FROM}/train_NER_augment_counter_and_rebuttal.csv')
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


LABEL_ALL_SUBTOKENS = True

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


        text = text.split()
        # TOKENIZE TEXT
        # is_split_into_words:假设输入已经按字切分，直接进行tokenize，适用于ner
        encoding = self.tokenizer(text,
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
IDS = train_text_df['id'].unique()
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

tokenizer = LongformerTokenizerFast.from_pretrained(DOWNLOADED_MODEL_PATH, add_prefix_space=True)
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

    def forward(self, input_ids, attn_masks):
        outputs = self.automodel(input_ids, attention_mask=attn_masks)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]),
                                  dim=-1)  # [bs, seq_len, hidden_dim*4]
        first_hidden_states = hidden_states[:, :, :]  # [bs, hidden_dim*4]
        logits = self.fc(first_hidden_states)
        return logits


#################################################################################
# 推理及验证数据
proba_thresh = {
    "Lead": 0.687,
    "Position": 0.537,
    "Evidence": 0.637,
    "Claim": 0.537,
    "Concluding Statement": 0.687,
    "Counterclaim": 0.537,
    "Rebuttal": 0.537,
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
model = MyModel(model_name=DOWNLOADED_MODEL_PATH, num_classes=15, freeze_bert=False)
model.to(config['device'])

def inference(batch):
    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])
    outputs = model(ids, attn_masks=mask)

    all_preds = torch.argmax(outputs, axis=-1).cpu().numpy()
    #twos = torch.argmax(is_two, axis=-1).cpu().numpy()
    outputs = torch.softmax(outputs, axis = -1).detach().cpu().numpy()
    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    sorces = []
    for k, text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        sorce = []
        word_ids = batch['wids'][k].numpy()
        previous_word_idx = word_ids[0]
        t = []
        down = []
        for idx, word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                # 用子词出现次数最多的标签作为原词预测标签

                if len(t) >= 1:
                    haha = max(t, key=t.count)
                    prediction.append(haha)
                    sorce.append(outputs[k][down[t.index(haha)]][text_preds[down[t.index(haha)]]])
                previous_word_idx = word_idx
                t, down = [], []
                t.append(token_preds[idx])
                down.append(idx)
                '''
                prediction.append(token_preds[idx])
                sorce.append(outputs[k][idx][text_preds[idx]])
                previous_word_idx = word_idx
                '''
            else:
                t.append(token_preds[idx])
                down.append(idx)

        haha = max(t, key=t.count)
        prediction.append(haha)
        sorce.append(outputs[k][down[t.index(haha)]][text_preds[down[t.index(haha)]]])

        predictions.append(prediction)
        sorces.append(sorce)

    return predictions, sorces


# 在推理阶段，对每个字词都进行预测。所以在最后得到输出标签时，要将这些自词合并
# TODO 尝试不同的子词标签合并方法

def get_predictions(df, loader):
    # put model in training mode

    # GET WORD LABEL PREDICTIONS
    global model
    final_pred1 = []
    ver = [1,3,4,5,6,7,8]
    for inn in ver:

        model = MyModel(model_name=DOWNLOADED_MODEL_PATH, num_classes=15, freeze_bert=False)
        model.to(config['device'])
        model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/longformer_v{inn}.pt'))
        model.eval()
        print(inn, ' loaded.')
        y_pred2 = []
        y_log = []
        for batch in loader:
            labels, logs = inference(batch)
            y_pred2.extend(labels)
            y_log.extend(logs)

        ka = []
        for i in range(len(df)):
            final_preds2 = []
            # pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
            pred = y_pred2[i]  # Leave "B" and "I"
            grade = y_log[i]
            j = 0
            torch.set_printoptions(threshold=np.inf)
            while j < len(pred):
                cls = pred[j]
                sum_prd = 0
                if cls != 'O':
                    cls = cls.replace('B', 'I')  # spans start with B
                    sum_prd += grade[j]
                end = j + 1
                while end < len(pred) and pred[end].replace('B', 'I') == cls:
                    sum_prd += grade[end]
                    end += 1
                if cls != 'O' and cls != '' and sum_prd/(end - j) >= proba_thresh[cls.replace('I-', '')] and end - j >= min_thresh[cls.replace('I-', '')]:
                    final_preds2 += [cls.replace('I-', '')]*(end - j)
                else:
                    final_preds2 += ['O']*(end - j)
                j = end
            if inn >= 5:
                final_preds2 += ['0']*(len(final_pred1[0][i]) - len(final_preds2))
            ka.append(final_preds2)
        torch.cuda.empty_cache()
        final_pred1.append(ka)
    final_pred = []
    for i in range(len(final_pred1[0])):
        sent = []
        for j in range(len(final_pred1[0][i])):
            ah = []
            for k in range(len(final_pred1)):
                ah.append(final_pred1[k][i][j])
            sent.append(max(ah, key=ah.count))
        final_pred.append(sent)
    final_pred3 = []
    for i in range(len(df)):

        idx = df.id.values[i]
        # pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
        pred = final_pred[i]
        # twoes = y_two[i]
        preds = []
        j = 0
        torch.set_printoptions(threshold=np.inf)
        while j < len(pred):
            cls = pred[j]
            # is_x = twoes[j]
            sum_prd = 0
            if cls != 'O':
                cls = cls.replace('B', 'I')  # spans start with B
            end = j + 1
            while end < len(pred) and pred[end].replace('B', 'I') == cls:
                end += 1

            if cls != 'O' and cls != '' and end - j >= min_thresh[cls]:
                final_pred3.append((idx, cls,
                                     ' '.join(map(str, list(range(j, end))))))
            j = end
    oof = pd.DataFrame(final_pred3)
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
