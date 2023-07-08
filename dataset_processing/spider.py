import csv
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from socket import error as SocketError
import errno
outcome = []
err = pd.read_csv('err_AERJE_dataset.csv')
er_id = list(err['id'])
with open('/url1.csv') as f:
    reader = csv.reader(f)
    result = list(reader)
    for u in er_id: #逐一获取进入他们的页面从而获取他的地址
        re_id = 'https://stackoverflow.com/questions/' + str(u)
        print(u)
        time.sleep(1)
        answer_list = []
        tags_ = []
        r = requests.get(re_id)
        r.encoding="utf-8"
        try:
            soup = BeautifulSoup(r.text,features="lxml").find('div',attrs={"class":"container"})
            soup1 =  BeautifulSoup(r.text,features="lxml").find('div',attrs={"id":"answers"}).find('div',attrs={"class":"post-layout"})
            title = soup.find('div',attrs={"id":"question-header"}).find('h1').text
            body = soup.find('div', attrs={"class": "postcell post-layout--right"}).find('div',attrs={"class":"s-prose js-post-body"}).find('p')
            newbody = re.findall('>(.*?)<',str(body))
            nbody = ''.join(newbody)
            year = soup.find("div",attrs={"class":"flex--item ws-nowrap mr16 mb8"}).find('time')
            newtime = re.findall(">(.*?)<",str(year))
            ntime = ''.join(newtime)
            tags = soup.find('div',attrs={"class":"mt24 mb12"}).find_all('a')
            for t in tags:
                tag = t.text
                tags_.append(tag)
            if soup1.find('div',attrs={"class":"votecell post-layout--left"}).find('div',attrs={"class":"ta-center"})!=None:
                answer = soup1.find('div',attrs={"class":"answercell post-layout--right"}).find_all('p')
                for item in answer:
                    item = str(item)
                    ans = re.findall('>(.*?)<', item)
                    newans=''.join(ans)
                    answer_list.append(newans)
                task = {"Title":None,"Tag":None,"Body":None,"Answer":None}
                task['Title'] = title
                task['Body'] = nbody
                task['Tag'] = tags_
                for i in answer:
                    answer_text = i.text
                    answer_list.append(answer_text)
                task['Answer'] = answer_list
                outcome.append(task)
            df = pd.DataFrame(outcome)
            df.to_csv('../dataset_processing/err_all_AERJE.csv', index=False)
        except AttributeError:
            pass

