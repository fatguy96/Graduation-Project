# coding:utf-8
import csv
import re
from bs4 import BeautifulSoup
from selenium import webdriver


def output_html(soup):

    time = ""
    weather = ""
    temperature = ""

    Weather_conditions=['小雨', '中雨', '大雨', '晴', '多云', '零散雷雨', '阴', '暴雨']
    rows = []
    i = 0
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
                    # print(content)
                j = j + 1
        i += 1

    with open('weather/K1167_470_weather.csv', 'a') as f:
        f_csv = csv.writer(f,)
        f_csv.writerows(rows)


urls = ['http://www.tianqihoubao.com/lishi/kunshan/month/201706.html',
        'http://www.tianqihoubao.com/lishi/kunshan/month/201707.html',
        'http://www.tianqihoubao.com/lishi/kunshan/month/201708.html']

for url in urls:
    driver = webdriver.PhantomJS(executable_path='/home/fate/Downloads/phantomjs-2.1.1-linux-x86_64/bin/phantomjs')
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser', from_encoding='utf-8')
    title = soup.find('div', class_='wdetail').find('table').find('tbody')
    output_html(title)










