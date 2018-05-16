# coding:utf-8
from urllib import request
from bs4 import BeautifulSoup
from selenium import webdriver
import re

url = 'http://www.tianqihoubao.com/lishi/kunshan/month/201706.html'
# response = request.urlopen(url)
# html = response.read()
# html = html.decode("gbk")
# soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
driver = webdriver.PhantomJS(executable_path='/home/fate/Downloads/phantomjs-2.1.1-linux-x86_64/bin/phantomjs')
driver.get(url)
soup = BeautifulSoup(driver.page_source, 'html.parser', from_encoding='utf-8')
title = soup.find('div', class_='wdetail').find('table').find('tbody')
i = 0
rows = []
for tr in soup.find_all('tr'):
    if i == 0:
        pass
    else:
        j = 0
        for td in tr.find_all('td'):

            if j == 0:
                a = td.find('a')
                time = re.sub('\s', '', a.string)
            if j == 1:
                weather = re.sub('\s', '', td.string)
            if j == 2:
                temperature = re.sub('\s', '', td.string)
            if j == 3:
                wind = re.sub('\s', '', td.string)
                content = (time, weather, temperature, wind)
                rows.append(content)
            j = j + 1
    i += 1
print(rows, end='\n')