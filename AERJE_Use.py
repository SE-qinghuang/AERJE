#!/usr/bin/env python
# -*- coding:utf-8 -*-
import ast
import csv
import json
import re
from numpy import *

import pandas as pd
from tqdm import tqdm
import transformers as huggingface_transformers
from AERJE.extraction.record_schema import RecordSchema
from AERJE.sel2record.record import MapConfig
from AERJE.extraction.scorer import *
from AERJE.sel2record.sel2record import SEL2Record
import math
import pandas as pd
import numpy as np
import json, time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
bert_path = "base_models/AERJE-generator-en/"  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer = BertTokenizer.from_pretrained(bert_path)

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
model = Bert_Model(bert_path).to(DEVICE)

model.load_state_dict(torch.load("./classifier_data/saved_bert_model/best_bert_model.pth"))

def classer(text):
    maxlen = 30
    # text = input("Input: ")
    encode_dict = tokenizer.encode_plus(text=text, max_length=100,
                                        padding='max_length', truncation=True)
    input_ids = encode_dict['input_ids']
    input_types = encode_dict['token_type_ids']
    input_masks = encode_dict['attention_mask']

    input_ids = torch.LongTensor(np.array(input_ids))
    input_ids = input_ids.unsqueeze(0)
    input_types = torch.LongTensor(np.array(input_types))
    input_types = input_types.unsqueeze(0)
    input_masks = torch.LongTensor(np.array(input_masks))
    input_masks = input_masks.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        y_pred = model(input_ids.to(DEVICE), input_masks.to(DEVICE), input_types.to(DEVICE))
        y_pred = torch.topk(y_pred, 3, dim=1)
        y_pred = y_pred[1][0].detach().cpu().numpy().tolist()

    relation1 = ['function similarity', 'behavior difference', 'function replace',
'logic constraint', 'efficiency comparison']
    pre_rel = [relation1[i] for i in y_pred]
    return pre_rel

split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
special_to_remove = {'<pad>', '</s>'}


def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name)]


def schema_to_ssi(schema: RecordSchema):
    ssi = "<spot> " + "<spot> ".join(sorted(schema.type_list))
    ssi += "<asoc> " + "<asoc> ".join(sorted(schema.role_list))
    ssi += "<extra_id_2> "
    return ssi


def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()


class HuggingfacePredictor:
    def __init__(self, model_path, schema_file, max_source_length=256, max_target_length=192) -> None:
        self._tokenizer = huggingface_transformers.T5TokenizerFast.from_pretrained(
            model_path,force_download = True)
        self._model = huggingface_transformers.T5ForConditionalGeneration.from_pretrained(model_path,force_download = True)
        self._model.cuda()
        self._schema = RecordSchema.read_from_file(schema_file)
        self._ssi = schema_to_ssi(self._schema)
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length

    def predict(self, text):
            pre = classer(text)
            pre_list = []
            for i in range(3):
                pre_list.append(pre[i])

            ssi = " <spot> " + " API "
            ssi += " <asoc> " + " <asoc> ".join(pre_list)
            ssi += " <extra_id_2> "
            self._ssi = ssi
            text = [self._ssi + text]
            inputs = self._tokenizer(
            text, padding=True, return_tensors='pt').to(self._model.device)

            inputs['input_ids'] = inputs['input_ids'][:, :self._max_source_length]
            inputs['attention_mask'] = inputs['attention_mask'][:,:self._max_source_length]

            result = self._model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self._max_target_length,
                num_beams=1,
                num_return_sequences = 1,
                return_dict_in_generate = True,
                output_scores = True
            )

            return self._tokenizer.batch_decode(result['sequences'], skip_special_tokens=False, clean_up_tokenization_spaces=False)

task_dict = {
    'entity': EntityScorer,
    'relation': RelationScorer,
}


def get_records(data_file,model_file):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', '-d', default=data_file)
    parser.add_argument(
        '--model', '-m', default=model_file)
    parser.add_argument('--max_source_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('-c', '--config', dest='map_config',
                        help='Offset Re-mapping Config',
                        default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--match_mode', default='normal',
                        choices=['set', 'normal', 'multimatch'])
    options = parser.parse_args()

    data_folder = options.data
    model_path = options.model

    predictor = HuggingfacePredictor(
        model_path=model_path,
        schema_file="./dataset_processing/secondwork/record.schema",
        max_source_length=options.max_source_length,
        max_target_length=options.max_target_length,
    )

    map_config = MapConfig.load_from_yaml(options.map_config)
    schema_dict = SEL2Record.load_schema_dict('./dataset_processing/secondwork')
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=options.decoding,
        map_config=map_config,
    )
    records = list()  ### collect entity and relation
    final_result = []
    text_list = []
    token_list = []
    data_list = []
    with open(data_file) as f:
        reader = csv.reader(f)
        result = list(reader)
        for i in range(len(result)):
            if result[i][0] not in data_list:
                data_list.append(result[i][0])
    for k in range(len(data_list)):
        text_list.append(data_list[k].lower())
        token_list.append(data_list[k].lower().split(" "))

    batch_num = math.ceil(len(text_list) / options.batch_size)

    predict_list = list()  ### collect SEL
    for index in tqdm(range(batch_num)):
        start = index * options.batch_size
        end = index * options.batch_size + options.batch_size

        pred_seq2seqs = [predictor.predict(s) for s in text_list[start: end]]
        pred_seq2seq = [post_processing(x[0]) for x in pred_seq2seqs] ### generate   SEL

        predict_list += pred_seq2seq

    final_predict_list = []
    for i in range(len(predict_list)):

        str = predict_list[i]
        str_1 = ((str.replace('  ','').replace(' ', '').replace('<extra_id_0>', '    <extra_id_0>    ').replace('<extra_id_1>',
                                                                                               '    <extra_id_1>    ').replace(
            '<extra_id_2>', '    <extra_id_2>    ').replace('<extra_id_3>', '    <extra_id_3>    ').replace(
            '<extra_id_4>',
            '    <extra_id_4>    ').replace(
            '<extra_id_5>', '    <extra_id_5>    ')).replace('    ', ' ')).replace("  ", ' ')
        str_new = ((' ' + str_1).replace('  ', '')).replace('functionsimilarity', 'function similarity').replace(
            'behaviordifference', 'behavior difference').replace('logicconstraint', 'logic constraint').replace(
            'efficiencycomparison', 'efficiency comparison').replace('functionreplace', 'function replace')
        final_predict_list.append(str_new)

    for p, text, tokens in zip(final_predict_list, text_list, token_list):
        task = {"sentence":None, "relation":None}
        relation = []
        r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
        if len(r.get('relation')['string']) == 2:
            relation.append(r.get('relation')['string'][0])
            relation.append(r.get('relation')['string'][1])
        if len(r.get('relation')['string']) == 1:
            relation.append(r.get('relation')['string'][0])
        task['sentence'] = text
        task['relation'] = relation
        final_result.append(task)
    df = pd.DataFrame(final_result)
    df.to_csv('./output/uie_result.csv',header=False,index=False)


if __name__ =='__main__':
    get_records('./pairs_5028.csv','./K=UIE/')
