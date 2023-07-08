import csv
import json
import os
import random

import pandas as pd

def convert_er(url1,url2):
    with open(url1,encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        result = list(reader)
        # random.shuffle(result)
        # print(result[1][3])
        out= []
        err = []
        j=0
        for i in range(1, len(result)):
            try:
                ### 实体1 ###
                entity_1 = result[i][0].lower()
                ### 关系1 ###
                relation1 = result[i][1].lower()
                ### 实体2 ###
                entity_2 = result[i][2].lower()
                ### 实体3 ###
                entity_3 = result[i][3].lower()
                ### 关系2 ###
                relation2 = result[i][4].lower()
                ### 实体4 ###
                entity_4 = result[i][5].lower()
                text = (result[i][6] + ' ').replace(". "," . ").replace(","," , ").replace("?"," ? ").lower()
                sen_token = text.split(" ")

                task = {"text":None,"tokens":None,"record":None,"entity":None,"relation":None,"event":None,"spot":None,"asoc":None,"spot_asoc":None}
                task['text'] = text.strip()
                task['tokens'] = sen_token

                entity_1_args = {"type": "API", "offset": None, "text": None}
                # entity_before['type'] = "Entity"
                entity_2_args = {"type": "API", "offset": None, "text": None}
                # entity_after['type'] = "Entity"
                entity_3_args = {"type": "API", "offset": None, "text": None}
                # entity_before['type'] = "Entity"
                entity_4_args = {"type": "API", "offset": None, "text": None}
                # entity_after['type'] = "Entity"


                relation1_arg = {"type": None, "args": None}
                relation2_arg = {"type": None, "args": None}

                spot_asoc_1 = {"span": None, "label": None, "asoc": None}
                spot_asoc_2 = {"span": None, "label": None, "asoc": None}
                spot_asoc_3 = {"span": None, "label": None, "asoc": None}
                spot_asoc_4 = {"span": None, "label": None, "asoc": None}

                task['spot'] = ['API']
                task['event'] = []

                if relation1!= "" and relation2!= "" :

                    entity_1_args['offset'] = [sen_token.index(entity_1)]
                    entity_1_args['text'] = entity_1
                    entity_2_args['offset'] = [sen_token.index(entity_2)]
                    entity_2_args['text'] = entity_2
                    entity_3_args['offset'] = [sen_token.index(entity_3)]
                    entity_3_args['text'] = entity_3
                    entity_4_args['offset'] = [sen_token.index(entity_4)]
                    entity_4_args['text'] = entity_4

                    task['record'] = "<extra_id_0> " \
                                     "<extra_id_0> API <extra_id_5> " + entity_1 + " <extra_id_0> " + relation1 + " <extra_id_5> " + entity_2 + " <extra_id_1> <extra_id_1> <extra_id_0> API <extra_id_5> " + entity_2 + " <extra_id_1> " + \
                                     "<extra_id_0> API <extra_id_5> " + entity_3 + " <extra_id_0> " + relation2 + " <extra_id_5> " + entity_4 + " <extra_id_1> <extra_id_1> <extra_id_0> API <extra_id_5> " + entity_4 + " <extra_id_1> " + \
                                     "<extra_id_1>"
                    task['entity'] = [entity_1_args, entity_2_args,entity_3_args,entity_4_args]

                    relation1_arg['type'] = relation1
                    relation1_arg['args'] = [entity_1_args, entity_2_args]

                    relation2_arg['type'] = relation2
                    relation2_arg['args'] = [entity_3_args, entity_4_args]

                    task['relation'] = [relation1_arg,relation2_arg]
                    task["asoc"] = [relation1,relation2]

                    spot_asoc_1['span'] = entity_1
                    spot_asoc_1['label'] = 'API'
                    spot_asoc_1['asoc'] = [[relation1, entity_2]]
                    spot_asoc_2['span'] = entity_2
                    spot_asoc_2['label'] = 'API'
                    spot_asoc_2['asoc'] = []
                    spot_asoc_3['span'] = entity_3
                    spot_asoc_3['label'] = 'API'
                    spot_asoc_3['asoc'] = [[relation2, entity_4]]
                    spot_asoc_4['span'] = entity_4
                    spot_asoc_4['label'] = 'API'
                    spot_asoc_4['asoc'] = []
                    task['spot_asoc'] = [spot_asoc_1, spot_asoc_2, spot_asoc_3,spot_asoc_4]

                elif relation1 !="" and relation2 =="":

                    relation1_arg['type'] = relation1
                    relation1_arg['args'] = [entity_1_args, entity_2_args]
                    task['relation'] = [relation1_arg]
                    task["asoc"] = [relation1]

                    if entity_3 != "":

                        entity_1_args['offset'] = [sen_token.index(entity_1)]
                        entity_1_args['text'] = entity_1
                        entity_2_args['offset'] = [sen_token.index(entity_2)]
                        entity_2_args['text'] = entity_2
                        entity_3_args['offset'] = [sen_token.index(entity_3)]
                        entity_3_args['text'] = entity_3

                        task['record'] = "<extra_id_0> " \
                                         "<extra_id_0> API <extra_id_5> " + entity_1 + " <extra_id_0> " + relation1 + " <extra_id_5> " + entity_2 + " <extra_id_1> <extra_id_1> <extra_id_0> API <extra_id_5> " + entity_2 + " <extra_id_1> " + \
                                         "<extra_id_0> API <extra_id_5> " + entity_3 + " <extra_id_1> " + \
                                         "<extra_id_1>"

                        task['entity'] = [entity_1_args, entity_2_args, entity_3_args]
                        spot_asoc_1['span'] = entity_1
                        spot_asoc_1['label'] = 'API'
                        spot_asoc_1['asoc'] = [[relation1, entity_2]]
                        spot_asoc_2['span'] = entity_2
                        spot_asoc_2['label'] = 'API'
                        spot_asoc_2['asoc'] = []
                        spot_asoc_3['span'] = entity_3
                        spot_asoc_3['label'] = 'API'
                        spot_asoc_3['asoc'] = []
                        task['spot_asoc'] = [spot_asoc_1, spot_asoc_2,spot_asoc_3]

                    if entity_3 == "":

                        entity_1_args['offset'] = [sen_token.index(entity_1)]
                        entity_1_args['text'] = entity_1
                        entity_2_args['offset'] = [sen_token.index(entity_2)]
                        entity_2_args['text'] = entity_2


                        task['record'] = "<extra_id_0>" \
                                         " <extra_id_0> API <extra_id_5> " + entity_1 + " <extra_id_0> " + relation1 + " <extra_id_5> " + entity_2 + " <extra_id_1> <extra_id_1> <extra_id_0> API <extra_id_5> " + entity_2 + " <extra_id_1> " +  \
                                         "<extra_id_1>"

                        task['entity'] = [entity_1_args, entity_2_args]

                        spot_asoc_1['span'] = entity_1
                        spot_asoc_1['label'] = 'API'
                        spot_asoc_1['asoc'] = [[relation1, entity_2]]
                        spot_asoc_2['span'] = entity_2
                        spot_asoc_2['label'] = 'API'
                        spot_asoc_2['asoc'] = []
                        task['spot_asoc'] = [spot_asoc_1,spot_asoc_2]


                elif relation1 == "":

                    entity_1_args['offset'] = [sen_token.index(entity_1)]
                    entity_1_args['text'] = entity_1

                    task['record'] = "<extra_id_0> " \
                                     "<extra_id_0> API <extra_id_5> " + entity_1 + " <extra_id_1> "+\
                                     "<extra_id_1>"
                    task['entity'] = [entity_1_args]
                    task['relation'] = []
                    task["asoc"] = []

                    spot_asoc_1['span'] = entity_1
                    spot_asoc_1['label'] = 'API'
                    spot_asoc_1['asoc'] = []
                    task['spot_asoc'] = [spot_asoc_1]



                out.append(task)
                # j += 1
                # if j == 2354:
                #     break

            except Exception as e:
                if i not in err:
                    err.append(str(i) + str(e))
                pass
        df = pd.DataFrame(err)
        df.to_csv('./try.csv',index=False)


        # print(len(out))
        # random.shuffle(out)

        for t in range(len(out)):
            d = json.dumps(out[t])
            with open(url2,'a',encoding='utf-8') as fw: # ../dataset_processing/converted_data/try_8_2/train.json'
                fw.write(d)
                fw.write('\n')

        # for v in range(int(0.6*len(out)),int(0.7*len(out))):
        #     d = json.dumps(out[v])
        #     with open('../data/API_ER/6_1_3_ER/val.json', 'a', encoding='utf-8') as fw:
        #         fw.write(d)
        #         fw.write('\n')

        # for te in range(int(0.8*len(out)),int(1*len(out))):
        #     d = json.dumps(out[te])
        #     with open('../dataset_processing/converted_data/try_8_2/test.json','a',encoding='utf-8') as fw:
        #         fw.write(d)
        #         fw.write('\n')




def convert_e(url,url2):
    with open(url, encoding='utf-8') as f:
        reader = csv.reader(f)
        result = list(reader)
        out = []
        for i in range(1, len(result)):
            try:
                ### 实体1 ###
                entity_1 = result[i][0].lower()
                ### 关系1 ###
                relation1 = ""
                ### 实体2 ###
                entity_2 = result[i][2].lower()
                ### 实体3 ###
                entity_3 = result[i][3].lower()
                ### 关系2 ###
                relation2 = ""
                ### 实体4 ###
                entity_4 = result[i][5].lower()
                text = (result[i][6] + ' ').replace(". ", " . ").replace(", ", " , ").replace("? ", " ? ").lower()
                sen_token = text.split(" ")

                task = {"text": None, "tokens": None, "record": None, "entity": None, "relation": None, "event": None,
                        "spot": None, "asoc": None, "spot_asoc": None}
                task['text'] = text
                task['tokens'] = sen_token

                entity_1_args = {"type": "API", "offset": None, "text": None}
                # entity_before['type'] = "Entity"
                entity_2_args = {"type": "API", "offset": None, "text": None}
                # entity_after['type'] = "Entity"
                entity_3_args = {"type": "API", "offset": None, "text": None}
                # entity_before['type'] = "Entity"
                entity_4_args = {"type": "API", "offset": None, "text": None}
                # entity_after['type'] = "Entity"

                # relation1_arg = {"type": None, "args": None}
                # relation2_arg = {"type": None, "args": None}

                spot_asoc_1 = {"span": None, "label": None, "asoc": None}
                spot_asoc_2 = {"span": None, "label": None, "asoc": None}
                spot_asoc_3 = {"span": None, "label": None, "asoc": None}
                spot_asoc_4 = {"span": None, "label": None, "asoc": None}

                task['spot'] = ['API']
                task['event'] = []

                if entity_4 != "":
                    entity_1_args['offset'] = [sen_token.index(entity_1)]
                    entity_1_args['text'] = entity_1
                    entity_2_args['offset'] = [sen_token.index(entity_2)]
                    entity_2_args['text'] = entity_2
                    entity_3_args['offset'] = [sen_token.index(entity_3)]
                    entity_3_args['text'] = entity_3
                    entity_4_args['offset'] = [sen_token.index(entity_4)]
                    entity_4_args['text'] = entity_4

                    task['record'] = "<extra_id_0>" \
                                     " <extra_id_0> API <extra_id_5>" + entity_1 + " <extra_id_1>" + \
                                     " <extra_id_0> API <extra_id_5>" + entity_2 + " <extra_id_1>" + \
                                     " <extra_id_0> API <extra_id_5>" + entity_3 + " <extra_id_1>" + \
                                     " <extra_id_0> API <extra_id_5>" + entity_4 + " <extra_id_1>" + \
                                     " <extra_id_1>"
                    task['entity'] = [entity_1_args, entity_2_args, entity_3_args, entity_4_args]

                    task['relation'] = []
                    task["asoc"] = []

                    spot_asoc_1['span'] = entity_1
                    spot_asoc_1['label'] = 'API'
                    spot_asoc_1['asoc'] = []

                    spot_asoc_2['span'] = entity_2
                    spot_asoc_2['label'] = 'API'
                    spot_asoc_2['asoc'] = []

                    spot_asoc_3['span'] = entity_3
                    spot_asoc_3['label'] = 'API'
                    spot_asoc_3['asoc'] = []

                    spot_asoc_4['span'] = entity_4
                    spot_asoc_4['label'] = 'API'
                    spot_asoc_4['asoc'] = []

                    task['spot_asoc'] = [spot_asoc_1, spot_asoc_2, spot_asoc_3, spot_asoc_4]


                elif entity_3 != "":
                    entity_1_args['offset'] = [sen_token.index(entity_1)]
                    entity_1_args['text'] = entity_1
                    entity_2_args['offset'] = [sen_token.index(entity_2)]
                    entity_2_args['text'] = entity_2
                    entity_3_args['offset'] = [sen_token.index(entity_3)]
                    entity_3_args['text'] = entity_3

                    task['record'] = "<extra_id_0>" \
                                     " <extra_id_0> API <extra_id_5> " + entity_1 + " <extra_id_1>" + \
                                     " <extra_id_0> API <extra_id_5> " + entity_2 + " <extra_id_1>" + \
                                     " <extra_id_0> API <extra_id_5> " + entity_3 + " <extra_id_1>" + \
                                     " <extra_id_1>"
                    task['entity'] = [entity_1_args, entity_2_args,entity_3_args]
                    task['relation'] = []
                    task["asoc"] = []

                    spot_asoc_1['span'] = entity_1
                    spot_asoc_1['label'] = 'API'
                    spot_asoc_1['asoc'] = []

                    spot_asoc_2['span'] = entity_2
                    spot_asoc_2['label'] = 'API'
                    spot_asoc_2['asoc'] = []

                    spot_asoc_3['span'] = entity_3
                    spot_asoc_3['label'] = 'API'
                    spot_asoc_3['asoc'] = []

                    task['spot_asoc'] = [spot_asoc_1, spot_asoc_2,spot_asoc_3]

                elif entity_2 != "":

                    entity_1_args['offset'] = [sen_token.index(entity_1)]
                    entity_1_args['text'] = entity_1
                    entity_2_args['offset'] = [sen_token.index(entity_2)]
                    entity_2_args['text'] = entity_2

                    task['record'] = "<extra_id_0>" \
                                     " <extra_id_0> API <extra_id_5> " + entity_1 + " <extra_id_1>" + \
                                     " <extra_id_0> API <extra_id_5> " + entity_2 + " <extra_id_1>" + \
                                     " <extra_id_1>"

                    task['entity'] = [entity_1_args, entity_2_args]
                    task['relation'] = []
                    task["asoc"] = []

                    spot_asoc_1['span'] = entity_1
                    spot_asoc_1['label'] = 'API'
                    spot_asoc_1['asoc'] = []

                    spot_asoc_2['span'] = entity_2
                    spot_asoc_2['label'] = 'API'
                    spot_asoc_2['asoc'] = []

                    task['spot_asoc'] = [spot_asoc_1, spot_asoc_2]

                elif entity_1 != "":

                    entity_1_args['offset'] = [sen_token.index(entity_1)]
                    entity_1_args['text'] = entity_1


                    task['record'] = "<extra_id_0>" \
                                     " <extra_id_0> API <extra_id_5> " + entity_1 + " <extra_id_1>" + \
                                     " <extra_id_1>"
                    task['entity'] = [entity_1_args]
                    task['relation'] = []
                    task["asoc"] = []

                    spot_asoc_1['span'] = entity_1
                    spot_asoc_1['label'] = 'API'
                    spot_asoc_1['asoc'] = []

                    task['spot_asoc'] = [spot_asoc_1]

            except Exception as e:
                with open('../data/try.txt', 'a') as f:
                    f.write(str(e))
                pass

            out.append(task)



        for t in range(int(len(out))):
            d = json.dumps(out[t])
            with open(url2, 'a', encoding='utf-8') as fw:
                fw.write(d)
                fw.write('\n')

        # for v in range(int(0.6 * len(out)), int(0.7 * len(out))):
        #     d = json.dumps(out[v])
        #     with open('../data/API_E/6_1_3_E/val.json', 'a', encoding='utf-8') as fw:
        #         fw.write(d)
        #         fw.write('\n')

        # for te in range(int(0.8 * len(out)), int(1 * len(out))):
        #     d = json.dumps(out[te])
        #     with open('../data/API_E/6_1_3_E/test.json', 'a', encoding='utf-8') as fw:
        #         fw.write(d)
        #         fw.write('\n')
def del_csv(url):
    with open(url,encoding='utf-8') as f:
        reader = csv.reader(f)
        result = list(reader)
        out =[]
        for i in result:
            if i[0]!="":
                task = {"api1":None,"relation1":None,"api2":None,"api3":None,"relation2":None,"api4":None,"sentence":None}
                task["api1"] = i[0]
                task["relation1"] = i[1]
                task["api2"] = i[2]
                task["api3"] = i[3]
                task["relation2"] = i[4]
                task["api4"] = i[5]
                task["sentence"] = i[6]
                out.append(task)
        random.shuffle(out)
        df =  pd.DataFrame(out)
        df.to_csv('../dataset_processing/1.csv',index=False)


def select(url):
    with open(url,encoding='utf-8') as f:
        reader = csv.reader(f)
        result = list(reader)
        out =[]
        delete = [17, 254, 260, 28, 339, 562, 145, 142, 127, 329, 212, 276]
        for i in range(1, len(result)):
            if i not in delete:
                task = {"api1":None,"relation1":None,"api2":None,"api3":None,"relation2":None,"api4":None,"sen":None}
                task["api1"] = result[i][0]
                task["relation1"] = result[i][1]
                task["api2"] = result[i][2]
                task["api3"] = result[i][3]
                task["relation2"] = result[i][4]
                task["api4"] = result[i][5]
                task["sen"] = result[i][6]
                out.append(task)
        # random.shuffle(out)
        df =  pd.DataFrame(out)
        df.to_csv('../dataset_processing/cy_test1.csv',index=False)

if __name__ == '__main__':
    # del_csv('../dataset_processing/increase_data.csv')
    # convert_e('/home/dell/SYB/UIE-main/dataset_processing/data/RQ2+3/test4uie.csv', '../dataset_processing/converted_data/RQ6/entity/test.json')
    # os.makedirs('../dataset_processing/converted_data/RQ5/nor')
    convert_er('../test4uie.csv','./test.json')
    # convert_er('../test4uie.csv', './test.json')
    # select('../dataset_processing/cy_test.csv')
    # convert_er('../dataset_processing/test4uie.csv','../dataset_processing/converted_data/RQ1+2/test.json')

    # for i in ['1o']:
    #     for j in range(1, 6):
    #         for s in ['f']:
    #             if i != j:
    #                 opt = "cross" + str(j)
    #                 # opt = j
    #                 out = '../dataset_processing/converted_data/RQ1/' + opt
    #
    #                 input_train_name = '../dataset_processing/data/cro_RQ2/' + opt + '/train.csv'
    #                 input_test_name = '../dataset_processing/data/cro_RQ2/' + opt + '/test4uie.csv'
    #                 output_trian_name = out + '/train.json'
    #                 output_test_name = out + '/test.json'
    #                 os.makedirs(out)
    #                 convert_er(input_train_name, output_trian_name)
    #                 convert_er(input_test_name, output_test_name)






