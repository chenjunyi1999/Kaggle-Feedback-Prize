import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
from torch import cuda
from sklearn.metrics import accuracy_score

"""
拿到代码先要干的事：
1、分段复制到kaggle notebook上
2、修改一下路径，看界面右边的kaggle文件夹
3、因为是离线比赛，添加kaggle提供的模型权重 我这个代码用的是"roberta-base"
4、kaggle notebook用完记得关 一个人一周只有40h

老张老李我爱你们！！！！！！！！！！！！！！！
"""

"""=====================================Data Preprocessing======================================"""
# pandas读测试集
test_names, test_texts = [], []
for f in tqdm(list(os.listdir('../input/feedback-prize-2021/test'))):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open('../input/feedback-prize-2021/test/' + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

# pandas读训练集
# meta data
train_df = pd.read_csv('../input/feedback-prize-2021/train.csv')
# 文本数据
train_names, train_texts = [], []
for f in tqdm(list(os.listdir('../input/feedback-prize-2021/train'))):
    train_names.append(f.replace('.txt', ''))
    train_texts.append(open('../input/feedback-prize-2021/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': train_names, 'text': train_texts})
# print(set(train_df['discourse_type'].tolist()))
# {'Counterclaim', 'Rebuttal', 'Position', 'Evidence', 'Claim', 'Concluding Statement', 'Lead'}

# 处理一下 给每个单词搭上标签 在标签前面加上B或I B表示一个类别的第一个，I表示后面的
all_entities = []
for idx, cur_train_text_df in tqdm(train_text_df.iterrows()):
    total = cur_train_text_df['text'].split(' ').__len__()
    start = -1
    entities = []
    # 从meta_data(train_df) 里给文本数据(train_text_df)找标签
    # ！！！！！！有的训练数据没有给出标签 这里统一设置成0
    for jdx, cur_train_df in train_df[train_df['id'] == cur_train_text_df['id']].iterrows():
        discourse = cur_train_df['discourse_type']
        list_ix = cur_train_df['predictionstring'].split(' ')
        ent = [f"I-{discourse}" for ix in list_ix]
        ent[0] = f"B-{discourse}"
        ds = int(list_ix[0])
        de = int(list_ix[-1])
        if start < ds - 1:
            ent_add = ['O' for ix in range(int(ds - 1 - start))]
            ent = ent_add + ent
        entities.extend(ent)
        start = de
    if len(entities) < total:
        ent_add = ["O" for ix in range(total - len(entities))]
        entities += ent_add
    else:
        entities = entities[:total]
    all_entities.append(entities)
train_text_df['entities'] = all_entities

"""=======================================label Mapping========================================="""
output_labels = ['O',
                 'B-Lead', 'I-Lead',
                 'B-Position', 'I-Position',
                 'B-Claim', 'I-Claim',
                 'B-Counterclaim', 'I-Counterclaim',
                 'B-Rebuttal', 'I-Rebuttal',
                 'B-Evidence', 'I-Evidence',
                 'B-Concluding Statement',
                 'I-Concluding Statement']

labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}

"""========================================Dataset=============================================="""


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 拿index对应的句子和标签列表
        sentence = self.data.text[index]
        word_labels = self.data.entities[index]

        # 用tokenizer处理(encoding padding truncation)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # 标签
        labels = [labels_to_ids[label] for label in word_labels]
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100 # 生成512的数组
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] != 0 and mapping[0] != encoding['offset_mapping'][idx - 1][1]:
                try:
                    encoded_labels[idx] = labels[i]
                except:
                    pass
                i += 1
            else:
                if idx == 1:
                    encoded_labels[idx] = labels[i]
                    i += 1

        # turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item


"""=================================Classification Model========================================"""
# todo 这里用的是Roberta ，RobertaForTokenClassification自带全连接分类，可以调用Roberta后面接自己的

config = {'model_name': '/kaggle/data/roberta-base/',
          'train_size': 0.8,
          'max_length': 512,
          'train_batch_size': 8,
          'valid_batch_size': 16,
          'epochs': 3,
          'learning_rate': 1e-05,
          'max_grad_norm': 10,
          'device': 'cuda' if cuda.is_available() else 'cpu'}

tokenizer = RobertaTokenizerFast.from_pretrained(config['model_name'])
model = RobertaForTokenClassification.from_pretrained(config['model_name'], num_labels=len(output_labels))

"""===================================train====================================================="""
# device
device = config['device']

# 准备数据
data = train_text_df[['text', 'entities']]
train_size = config['train_size']
train_dataset = data.sample(frac=train_size, random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
# dataset
training_set = dataset(train_dataset, tokenizer, config['max_length'])
testing_set = dataset(test_dataset, tokenizer, config['max_length'])
# dataloader
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 1,
                'pin_memory': True
                }
test_params = {'batch_size': config['valid_batch_size'],
               'shuffle': True,
               'num_workers': 1,
               'pin_memory': True
               }
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# model and optimizer
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])


def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    for idx, batch in enumerate(training_loader):

        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

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


for epoch in range(config['epochs']):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)


def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                      return_dict=False)

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions


labels, predictions = valid(model, testing_loader)

"""=============================inference and submission========================================"""
model.eval()


def inference(sentence):
    inputs = tokenizer(sentence,
                       return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True,
                       max_length=config['max_length'],
                       return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask, return_dict=False)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits,
                                         axis=1)  # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    out_str = []
    off_list = inputs["offset_mapping"].squeeze().tolist()
    for idx, mapping in enumerate(off_list):

        if mapping[0] != 0 and mapping[0] != off_list[idx - 1][1]:

            prediction.append(wp_preds[idx][1])
            out_str.append(wp_preds[idx][0])
        else:
            if idx == 1:
                prediction.append(wp_preds[idx][1])
                out_str.append(wp_preds[idx][0])
            continue
    return prediction, out_str


y_pred = []
for i, t in enumerate(test_texts['text'].tolist()):
    o,o_t = inference(t)
    y_pred.append(o)

final_preds = []
for i in tqdm(range(len(test_texts))):
    idx = test_texts.id.values[i]
    pred = [x.replace('B-', '').replace('I-', '') for x in y_pred[i]]
    preds = []
    j = 0
    while j < len(pred):
        cls = pred[j]
        if cls == 'O':
            j += 1
        end = j + 1
        while end < len(pred) and pred[end] == cls:
            end += 1
        if cls != 'O' and cls != '' and end - j > 10:
            final_preds.append((idx, cls, ' '.join(map(str, list(range(j, end))))))
        j = end
print(final_preds[1])


test_df = pd.read_csv('../data/feedback-prize-2021/sample_submission.csv')
sub = pd.DataFrame(final_preds)
sub.columns = test_df.columns
sub.to_csv("submission.csv", index=False)