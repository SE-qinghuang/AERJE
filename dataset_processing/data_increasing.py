import ast
import csv
import json
import os
import random
import re

import pandas as pd
import spacy
from numpy import *
# divide data
def divide_data():
    with open('/pure_data.csv') as f:
        reader = csv.reader(f)
        result = list(reader)
        out1 = []
        out2 = []
        for i in range(1,int(0.8*len(result))):
            task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None, "api4": None,"sentence": None}
            task['api1'] = result[i][0]
            task['relation1'] = result[i][1]
            task['api2'] = result[i][2]
            task['api3'] = result[i][3]
            task['relation2'] = result[i][4]
            task['api4'] = result[i][5]
            task['sentence'] = result[i][6]
            out1.append(task)
        for i in range(int(0.8 * len(result)),len(result)):
            task1 = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None, "api4": None,"sentence": None}
            task1['api1'] = result[i][0]
            task1['relation1'] = result[i][1]
            task1['api2'] = result[i][2]
            task1['api3'] = result[i][3]
            task1['relation2'] = result[i][4]
            task1['api4'] = result[i][5]
            task1['sentence'] = result[i][6]
            out2.append(task1)
        df1 = pd.DataFrame(out1)
        df1.to_csv('../dataset_processing/train.csv',index=False)

        df2 = pd.DataFrame(out2)
        df2.to_csv('../dataset_processing/test.csv',index=False)

def deal_space():
    with open('./1.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        result = list(reader)
        out_ = []
        for i in range(1, len(result)):
            task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None, "api4": None,"sen": None}
            task['api1'] = result[i][0]
            task['relation1'] = result[i][1]
            task['api2'] = result[i][2]
            task['api3'] = result[i][3]
            task['relation2'] = result[i][4]
            task['api4'] =result[i][5]
            task['sen'] = result[i][6].strip()
            out_.append(task)
        df = pd.DataFrame(out_)
        df.to_csv('./2.csv',index=False)

def data_increasing_api(url1,url2):
    with open(url1,encoding='utf-8') as f: #'./pure_data.csv'
        reader = csv.reader(f)
        result = list(reader)
        out_ =[]
        for i in range(1,len(result)):
            print(i)
            api_list = []
            api1_list = []
            api2_list = []
            api3_list = []
            api4_list = []
            relation_1 = result[i][1]
            relation_2 = result[i][4]
            # tag = result[i][7]
            # id = result[i][8]
            sen = (' '+result[i][6]+' ').replace(". ", " . ").replace(", ", " , ").replace("? ", " ? ").lower()
            # sen = ' '+result[i][6]+' '
            # sen_list = sen.split("")
            tag = result[i][7]
            label1 = result[i][8]
            label2 = result[i][9]
            CHECK = result[i][10]
            api1 = result[i][0]
            if api1!= "":
                api_list.append(api1)
            api2 = result[i][2]
            if api2 != "":
                api_list.append(api2)
            api3 = result[i][3]
            if api3 != "":
                api_list.append(api3)
            api4 = result[i][5]
            if api4 != "":
                api_list.append(api4)
            if api1 != "" and api2 !="" and api3 !="" and api4 != "":
                for k in range(len(api_list)):
                    if k == 0:
                        if "." in api_list[0] and "(" in api_list[0]:
                            newapi1  = api_list[0].split('.')[-1]
                            api1_list.append(newapi1)
                            newapi2  = re.sub('\(.*?\)','',newapi1)
                            api1_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)','',api_list[0])
                            api1_list.append(newapi3)
                        elif "." in api_list[0]:
                            newapi4 = api_list[0].split('.')[-1]
                            api1_list.append(newapi4)
                        elif "#" in api_list[0] and "(" in api_list[0]:
                            newapi6  = api_list[0].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)','',api_list[0])
                            api1_list.append(newapi8)
                        elif "#" in api_list[0]:
                            newapi9 = api_list[0].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[0]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi5)
                    if k == 1:
                        if "." in api_list[1] and "(" in api_list[1]:
                            newapi1 = api_list[1].split('.')[-1]
                            api2_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api2_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[1])
                            api2_list.append(newapi3)
                        elif "." in api_list[1]:
                            newapi4 = api_list[1].split('.')[-1]
                            api2_list.append(newapi4)
                        elif "#" in api_list[1] and "(" in api_list[1]:
                            newapi6  = api_list[1].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)','',api_list[1])
                            api1_list.append(newapi8)
                        elif "#" in api_list[1]:
                            newapi9 = api_list[1].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[1]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[1])
                            api2_list.append(newapi5)
                    if k == 2:
                        if "." in api_list[2] and "(" in api_list[2]:
                            newapi1 = api_list[2].split('.')[-1]
                            api3_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api3_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[2])
                            api3_list.append(newapi3)
                        elif "." in api_list[2]:
                            newapi4 = api_list[2].split('.')[-1]
                            api3_list.append(newapi4)
                        elif "#" in api_list[2] and "(" in api_list[2]:
                            newapi6  = api_list[2].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)','',api_list[2])
                            api1_list.append(newapi8)
                        elif "#" in api_list[2]:
                            newapi9 = api_list[2].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[2]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[2])
                            api3_list.append(newapi5)
                    if k == 3:
                        if "." in api_list[3] and "(" in api_list[3]:
                            newapi1 = api_list[3].split('.')[-1]
                            api4_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api4_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[3])
                            api4_list.append(newapi3)
                        elif "." in api_list[3]:
                            newapi4 = api_list[3].split('.')[-1]
                            api4_list.append(newapi4)
                        elif "#" in api_list[3] and "(" in api_list[3]:
                            newapi6  = api_list[3].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)','',api_list[3])
                            api1_list.append(newapi8)
                        elif "#" in api_list[3]:
                            newapi9 = api_list[3].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[3]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[3])
                            api4_list.append(newapi5)
                for a in range(len(api1_list)):
                    for b in range(len(api2_list)):
                        for c in range(len(api3_list)):
                            for d in range(len(api4_list)):
                                task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None, "api4": None,"sen": None, "tag": None, "id": None}
                                task['api1'] = api1_list[a]
                                task['relation1'] = relation_1
                                task['api2'] = api2_list[b]
                                task['api3'] = api3_list[c]
                                task['relation2'] = relation_2
                                task['api4'] = api4_list[d]
                                task['sen'] = sen.replace(' '+ api1+' ' ,' '+ api1_list[a] + ' ').replace(' ' + api2 +' ',' '+api2_list[b]+' ').replace(' ' + api3 + ' ',' '+ api3_list[c]+' ').replace(' '+api4+' ',' '+api4_list[d]+' ').strip()
                                # task['tag'] = tag
                                # task['id'] = id
                                task["tag"] = tag
                                task["label1"] = label1
                                task["label2"] = label2
                                task["CHECK"] = CHECK
                                if task not in out_:
                                    out_.append(task)
            elif api1 != "" and api2 !="" and api3 !="":
                for k in range(len(api_list)):
                    if k == 0:
                        if "." in api_list[0] and "(" in api_list[0]:
                            newapi1 = api_list[0].split('.')[-1]
                            api1_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api1_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi3)
                        elif "." in api_list[0]:
                            newapi4 = api_list[0].split('.')[-1]
                            api1_list.append(newapi4)
                        elif "#" in api_list[0] and "(" in api_list[0]:
                            newapi6 = api_list[0].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi8)
                        elif "#" in api_list[0]:
                            newapi9 = api_list[0].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[0]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi5)
                    if k == 1:
                        if "." in api_list[1] and "(" in api_list[1]:
                            newapi1 = api_list[1].split('.')[-1]
                            api2_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api2_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[1])
                            api2_list.append(newapi3)
                        elif "." in api_list[1]:
                            newapi4 = api_list[1].split('.')[-1]
                            api2_list.append(newapi4)
                        elif "#" in api_list[1] and "(" in api_list[1]:
                            newapi6 = api_list[1].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)', '', api_list[1])
                            api1_list.append(newapi8)
                        elif "#" in api_list[1]:
                            newapi9 = api_list[1].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[1]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[1])
                            api2_list.append(newapi5)
                    if k == 2:
                        if "." in api_list[2] and "(" in api_list[2]:
                            newapi1 = api_list[2].split('.')[-1]
                            api3_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api3_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[2])
                            api3_list.append(newapi3)
                        elif "." in api_list[2]:
                            newapi4 = api_list[2].split('.')[-1]
                            api3_list.append(newapi4)
                        elif "#" in api_list[2] and "(" in api_list[2]:
                            newapi6 = api_list[2].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)', '', api_list[2])
                            api1_list.append(newapi8)
                        elif "#" in api_list[2]:
                            newapi9 = api_list[2].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[2]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[2])
                            api3_list.append(newapi5)

                for a in range(len(api1_list)):
                    for b in range(len(api2_list)):
                        for c in range(len(api3_list)):
                            # for d in range(len(api4_list)):
                                task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None, "api4": None,"sen": None}
                                task['api1'] = api1_list[a]
                                task['relation1'] = relation_1
                                task['api2'] = api2_list[b]
                                task['api3'] = api3_list[c]
                                task['relation2'] = relation_2
                                task['api4'] = api4
                                task['sen'] = sen.replace(' '+api1+' ',' '+api1_list[a]+' ').replace(' '+api2+' ',' '+api2_list[b]+' ').replace(' '+api3+' ',' '+api3_list[c]+' ').strip()
                                # task['tag'] = tag
                                # task['id'] = id
                                # task['title'] = title
                                task["tag"] = tag
                                task["label1"] = label1
                                task["label2"] = label2
                                task["CHECK"] = CHECK
                                if task not in out_:
                                    out_.append(task)
            elif api1 != "" and api2 !="":
                for k in range(len(api_list)):
                    if k == 0:
                        if "." in api_list[0] and "(" in api_list[0]:
                            newapi1 = api_list[0].split('.')[-1]
                            api1_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api1_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi3)
                        elif "." in api_list[0]:
                            newapi4 = api_list[0].split('.')[-1]
                            api1_list.append(newapi4)
                        elif "#" in api_list[0] and "(" in api_list[0]:
                            newapi6 = api_list[0].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi8)
                        elif "#" in api_list[0]:
                            newapi9 = api_list[0].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[0]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi5)
                    if k == 1:
                        if "." in api_list[1] and "(" in api_list[1]:
                            newapi1 = api_list[1].split('.')[-1]
                            api2_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api2_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[1])
                            api2_list.append(newapi3)
                        elif "." in api_list[1]:
                            newapi4 = api_list[1].split('.')[-1]
                            api2_list.append(newapi4)
                        elif "#" in api_list[1] and "(" in api_list[1]:
                            newapi6 = api_list[1].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)', '', api_list[1])
                            api1_list.append(newapi8)
                        elif "#" in api_list[1]:
                            newapi9 = api_list[1].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[1]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[1])
                            api2_list.append(newapi5)

                for a in range(len(api1_list)):
                    for b in range(len(api2_list)):
                        # for c in range(len(api3_list)):
                            # for d in range(len(api4_list)):
                                task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None, "api4": None,"sen": None}
                                task['api1'] = api1_list[a]
                                task['relation1'] = relation_1
                                task['api2'] = api2_list[b]
                                task['api3'] = api3
                                task['relation2'] = relation_2
                                task['api4'] = api4
                                task['sen'] = sen.replace(' ' + api1+' ',' '+api1_list[a]+' ').replace(' '+api2+' ',' '+api2_list[b]+' ').strip()
                                # task['tag'] = tag
                                # task['id'] = id
                                # task['title'] = title
                                task["tag"] = tag
                                task["label1"] = label1
                                task["label2"] = label2
                                task["CHECK"] = CHECK
                                if task not in out_:
                                    out_.append(task)
            elif api1 != "":
                for k in range(len(api_list)):
                    if k == 0:
                        if "." in api_list[0] and "(" in api_list[0]:
                            newapi1 = api_list[0].split('.')[-1]
                            api1_list.append(newapi1)
                            newapi2 = re.sub('\(.*?\)', '', newapi1)
                            api1_list.append(newapi2)
                            newapi3 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi3)
                        elif "." in api_list[0]:
                            newapi4 = api_list[0].split('.')[-1]
                            api1_list.append(newapi4)
                        elif "#"  in api_list[0] and "(" in api_list[0]:
                            newapi6 = api_list[0].split('#')[-1]
                            api1_list.append(newapi6)
                            newapi7 = re.sub('\(.*?\)', '', newapi6)
                            api1_list.append(newapi7)
                            newapi8 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi8)
                        elif "#" in api_list[0]:
                            newapi9 = api_list[0].split('#')[-1]
                            api1_list.append(newapi9)
                        elif "(" in api_list[0]:
                            newapi5 = re.sub('\(.*?\)', '', api_list[0])
                            api1_list.append(newapi5)
                for a in range(len(api1_list)):
                    # for b in range(len(api2_list)):
                        # for c in range(len(api3_list)):
                            # for d in range(len(api4_list)):
                                task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None, "api4": None,"sen": None}
                                task['api1'] = api1_list[a]
                                task['relation1'] = relation_1
                                task['api2'] = api2
                                task['api3'] = api3
                                task['relation2'] = relation_2
                                task['api4'] = api4
                                task['sen'] = sen.replace(' '+api1+' ',' '+api1_list[a]+' ').strip()
                                # task['tag'] = tag
                                # task['id'] = id
                                # task['title'] = title
                                task["tag"] = tag
                                task["label1"] = label1
                                task["label2"] = label2
                                task["CHECK"] = CHECK
                                if task not in out_:
                                    out_.append(task)

            # print(out_)task['api2']
            df = pd.DataFrame(out_)
            df.to_csv(url2,index=False)

def data_increase_verb(url1,url2):
    out = []
    with open(url1, encoding='utf-8') as f:
        with open('./verb.csv', encoding='utf-8') as f1:
            reader1 = csv.reader(f1)
            result1 = list(reader1)
            reader = csv.reader(f)
            result = list(reader)
            api_list = []
            for i in range(1, len(result)):
                api1_ = result[i][0]
                relation1_ = result[i][1]
                api2_ = result[i][2]
                api3_ = result[i][3]
                relation2_ = result[i][4]
                api4_ = result[i][5]
                sentence = result[i][6]
                sentence_list = sentence.split(' ')
                # tag = result[i][7]
                # id = result[i][8]
                tag = result[i][7]
                label1 = result[i][8]
                label2 = result[i][9]
                CHECK = result[i][10]
                candi = []
                print(i)
                sen = (re.sub('\(.*?\)', '', result[i][6]))
                api1 = (re.sub('\(.*?\)', '', result[i][0]))
                api2 = (re.sub('\(.*?\)', '', result[i][2]))
                api3 = (re.sub('\(.*?\)', '', result[i][3]))
                api4 = (re.sub('\(.*?\)', '', result[i][5]))
                if api1 != "":
                    api_list.append(api1)
                if api2 != "":
                    api_list.append(api2)
                if api3 != "":
                    api_list.append(api3)
                if api4 != "":
                    api_list.append(api4)
                nlp = spacy.load('en_core_web_sm')
                doc = nlp(re.sub('\(.*?\)', '', sen))

                for token in doc:
                    for api in api_list:
                        if api == token.text and token.head.pos_ == "VERB" and token.head.text not in api_list:
                            replace_word = token.head.text
                            for j in range(1, len(result1)):
                                verb_list = ast.literal_eval(result1[j][0])
                                if token.head.text in verb_list:
                                    for k in verb_list:
                                        if k not in candi: ###
                                            candi.append(k)
                if  candi != []:
                    for l in range(len(candi)):
                        sen_ = []
                        task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None,
                                "api4": None, "sentence": None}
                        task['api1'] = api1_
                        task['relation1'] = relation1_
                        task['api2'] = api2_
                        task['api3'] = api3_
                        task['relation2'] = relation2_
                        task['api4'] = api4_
                        for n in sentence_list:
                            if n != replace_word:
                                sen_.append(n)
                            else:
                                sen_.append(candi[l])
                        final_sen = " ".join(sen_)
                        task['sentence'] = final_sen
                        # task['tag'] = tag
                        # task['id'] = id
                        task["tag"] = tag
                        task["label1"] = label1
                        task["label2"] = label2
                        task["CHECK"] = CHECK
                        if task not in out:
                            out.append(task)
            df = pd.DataFrame(out)
            df.to_csv(url2, index=False)

def data_increase_verb2():
    out = []
    with open('./increase_api.csv', encoding='utf-8') as f:
        with open('./verb.csv', encoding='utf-8') as f1:
            reader1 = csv.reader(f1)
            result1 = list(reader1)
            reader = csv.reader(f)
            result = list(reader)
            api_list = []
            for i in range(1, len(result)):
                api1_ = result[i][0]
                relation1_ = result[i][1]
                api2_ = result[i][2]
                api3_ = result[i][3]
                relation2_ = result[i][4]
                api4_ = result[i][5]
                sentence = result[i][6]
                sentence_list = sentence.split(' ')

                candi = []
                print(i)
                sen = (re.sub('\(.*?\)', '', result[i][6]))
                api1 = (re.sub('\(.*?\)', '', result[i][0]))
                api2 = (re.sub('\(.*?\)', '', result[i][2]))
                api3 = (re.sub('\(.*?\)', '', result[i][3]))
                api4 = (re.sub('\(.*?\)', '', result[i][5]))
                if api1 != "":
                    api_list.append(api1)
                if api2 != "":
                    api_list.append(api2)
                if api3 != "":
                    api_list.append(api3)
                if api4 != "":
                    api_list.append(api4)
                nlp = spacy.load('en_core_web_sm')
                doc = nlp(re.sub('\(.*?\)', '', sen))

                for token in doc:
                    for api in api_list:
                        if api == token.text and token.head.pos_ == "VERB" and token.head.text not in api_list:
                            replace_word = token.head.text
                            for j in range(1, len(result1)):
                                verb_list = ast.literal_eval(result1[j][0])
                                if token.head.text in verb_list:
                                    for k in verb_list:
                                        if k not in candi:
                                            candi.append(k)
                if candi != []:
                    for l in range(len(candi)):
                        sen_ = []
                        task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None,
                                "api4": None, "sentence": None}
                        task['api1'] = api1_
                        task['relation1'] = relation1_
                        task['api2'] = api2_
                        task['api3'] = api3_
                        task['relation2'] = relation2_
                        task['api4'] = api4_
                        for n in sentence_list:
                            if n != replace_word:
                                sen_.append(n)
                            else:
                                sen_.append(candi[l])
                        final_sen = " ".join(sen_)
                        task['sentence'] = final_sen
                        if task not in out:
                            out.append(task)
            df = pd.DataFrame(out)
            df.to_csv('./increase_verb2.csv', index=False)

def combine_file(url1,url2,url3,url4):
    out = []
    with open(url1) as f1:
        with open(url2) as f2:
            with open(url3) as f3:
                reader1 = csv.reader(f1)
                result1 = list(reader1)
                reader2 = csv.reader(f2)
                result2 = list(reader2)
                reader3 = csv.reader(f3)
                result3 = list(reader3)
                file_list = [result1,result2,result3]
                for k in range(len(file_list)):
                    count = 0
                    count2 = 0
                    for i in range(1,len(file_list[k])):
                        task = {"api1": None, "relation1": None, "api2": None, "api3": None, "relation2": None,"api4": None, "sentence": None}
                        task['api1'] = file_list[k][i][0]
                        task['relation1'] = file_list[k][i][1]
                        task['api2'] = file_list[k][i][2]
                        task['api3'] = file_list[k][i][3]
                        task['relation2'] = file_list[k][i][4]
                        task['api4'] = file_list[k][i][5]
                        task['sentence'] = file_list[k][i][6]
                        # task['tag'] = file_list[k][i][7]
                        # task['id'] = file_list[k][i][8]
                        task["tag"] = file_list[k][i][7]
                        task["label1"] = file_list[k][i][8]
                        task["label2"] = file_list[k][i][9]
                        task["CHECK"] = file_list[k][i][10]
                        if task not in out:
                            if task['relation1'] != "":
                                count2 += 1
                            count += 1
                            out.append(task)
                    print(count)
                    print(count2)
                # random.shuffle(out)
                df = pd.DataFrame(out)
                df.to_csv(url4,index=False)

def ralation4class(url1,url2):
    with open(url1) as f:
        reader = csv.reader(f)
        result = list(reader)
        out = []
        relation_all = ['function similarity', 'behavior difference', 'function replace', 'function collaboration','type conversion', 'logic constraint', 'efficiency comparison']
        # i = 0
        # if 'train' in url1:
        #     n = 00
        # if 'test' in url1:
        #     n = 100
        for i in range(1,len(result)):
            if result[i][1]!= "":
                if result[i][4] == "":
                    task = {"sentence": None,"rel":None}
                    task['sentence'] = result[i][6]
                    task['rel'] = relation_all.index(result[i][1])
                    out.append(task)
                if result[i][4] != "":
                    task1 = {"sentence": None, "rel": None}
                    task1['sentence'] = result[i][6]
                    task1['rel'] = relation_all.index(result[i][1])
                    out.append(task1)
                    task2 = {"sentence": None, "rel": None}
                    task2['sentence'] = result[i][6]
                    task2['rel'] = relation_all.index(result[i][4])
                    out.append(task2)
            # if result[i][1] == "" :
            #     task = {"sentence": None, "rel": None}
            #     task['sentence'] = result[i][6]
            #     task['rel'] = 0
            #     out.append(task)
            #     i = i + 1
        random.shuffle(out)
        df = pd.DataFrame(out)
        df.to_csv(url2,index=False,header=False)





if __name__ =="__main__":
    # divide_data()
    # data_increasing_api('/home/dell/SYB/UIE-main/dataset_processing/cy_train.csv', '../dataset_processing/train_api.csv')
    # data_increase_verb('/home/dell/SYB/UIE-main/dataset_processing/cy_train.csv', '../dataset_processing/train_verb.csv')
    # combine_file('/home/dell/SYB/UIE-main/dataset_processing/cy_train.csv', '../dataset_processing/train_api.csv', '../dataset_processing/train_verb.csv', '../dataset_processing/data/RQ2+3/train4uie.csv')

    for i in ['1o']:
        for j in range(1):
            for s in [ ['util.csv', 'util_train.csv']]:
                if i != j:
                    opt = ''
                    input = '/home/dell/SYB/UIE-main/dataset_processing/data/RQ4_TF/' + opt  + s[0]
                    output = '/home/dell/SYB/UIE-main/dataset_processing/data/RQ4_TF/' + opt + s[1]
                    data_increasing_api(input, '../dataset_processing/test_api.csv')
                    data_increase_verb(input,'../dataset_processing/test_verb.csv')
                    combine_file(input,'../dataset_processing/test_api.csv','../dataset_processing/test_verb.csv',output)
                    # os.makedirs('../classifier_data/cor_RQ2/' + opt )
                    # ralation4class(output,'../classifier_data/cor_RQ2/' + opt + '/test4class.csv')
                    # ralation4class('../dataset_processing/data/RQ2+3/train4uie.csv','../classifier_data/train4class3.csv')
                    # ralation4class('../dataset_processing/data/RQ2+3/test4uie.csv','../classifier_data/test4class3.csv')
