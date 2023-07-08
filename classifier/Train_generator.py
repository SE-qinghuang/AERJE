import csv

import pandas as pd
import numpy as np
import json, time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

def get_data(url):
    input_ids, input_masks, input_types, = [], [], []  # input char ids, segment type ids,  attention mask
    labels = []  # 标签
    maxlen = 100  # 取100
    with open(url, encoding='utf-8') as f:
        reader  = csv.reader(f)
        result = list(reader)
        for i in range(len(result)):
            title = result[i][0]
            y = result[i][1]

            # encode_plus会输出一个字典，分别为'input_ids', 'token_type_ids', 'attention_mask'对应的编码
            # 根据参数会短则补齐，长则切断
            encode_dict = tokenizer.encode_plus(text=title, max_length=maxlen,padding='max_length', truncation=True)

            input_ids.append(encode_dict['input_ids'])
            input_types.append(encode_dict['token_type_ids'])
            input_masks.append(encode_dict['attention_mask'])

            labels.append(int(y))

        input_ids, input_types, input_masks = np.array(input_ids), np.array(input_types), np.array(input_masks)
        labels = np.array(labels)
        return input_ids,input_types,input_masks,labels

class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=5):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool)  # [bs, classes]
        return logit

def get_parameter_number(model):
    #  打印模型参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)


# 测试集没有标签，需要预测提交
def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
    return val_pred


def train_model(model, train_loader,
                   optimizer, scheduler, device, epoch):

    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        """训练模型"""
        start = time.time()
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            y_pred = model(ids, att, tpe)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率变化

            train_loss_sum += loss.item()
            # if (idx + 1) % (len(train_loader) // 5) == 0:  # 只打印五次结果
            print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} ".format(
                    i + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1)))

        torch.save(model.state_dict(), "../classifier_data/save_bert_model/")

if __name__ =='__main__':

    bert_path = "../base_models/AERJE-generator-en/"  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
    tokenizer = BertTokenizer.from_pretrained(bert_path)  # 初始化分词器
    BATCH_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
    model = Bert_Model(bert_path).to(DEVICE)
    # print(get_parameter_number(model))

    input_ids_train,input_types_train,input_masks_train,y_train = get_data('../classifier_data/train4class.csv')
    input_ids_test,input_types_test,input_masks_test,y_test = get_data('../classifier_data/test4class.csv')

    # 训练集
    train_data = TensorDataset(torch.LongTensor(input_ids_train),
                               torch.LongTensor(input_masks_train),
                               torch.LongTensor(input_types_train),
                               torch.LongTensor(y_train))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # 测试集（是没有标签的）
    test_data = TensorDataset(torch.LongTensor(input_ids_test),
                              torch.LongTensor(input_masks_test),
                              torch.LongTensor(input_types_test))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                                num_training_steps=EPOCHS * len(train_loader))

    train_model(model, train_loader, optimizer, scheduler, DEVICE, EPOCHS)

    # 加载最优权重对测试集测试
    model.load_state_dict(torch.load("../classifier_data/save_bert_model/best_bert_model.pth"))
    pred_test = predict(model, test_loader, DEVICE)
    print("\n Test Accuracy = {} \n".format(accuracy_score(y_test, pred_test)))
    print(classification_report(y_test, pred_test, digits=4))