import csv
import re

csv_reader = csv.reader(open('weather/K1135_20_weather.csv'))

Weather_conditions = ['无', '晴', '多云', '阴', '零散雷雨', '小雨', '中雨', '大雨', '暴雨']

rows = []
for row in csv_reader:
    weather = row[1].split('/')
    wind = row[3].split('/')
    first_wi = int(re.sub('\D', "", wind[0]))
    second_wi = int(re.sub('\D', "", wind[1]))

    if first_wi > 10:
        decade = int(first_wi/10)
        bits = first_wi - decade*10
        first_wi = (decade + bits) / 2

    if second_wi > 10:
        decade = int(second_wi / 10)
        bits = second_wi - decade * 10
        second_wi = (decade + bits) / 2

    winds = first_wi*10 + second_wi

    if weather[0] in Weather_conditions and weather[1] in Weather_conditions:
        first_we = Weather_conditions.index(weather[0])
        second_we = Weather_conditions.index(weather[1])
    else:
        first_we = second_we = 0
    condition = first_we*10 + second_we
    rows.append([row[0], condition, winds])
    print([row[0], condition, winds])

with open('weather/K1135_20_weather1.csv', 'w+') as f:
    f_csv = csv.writer(f,)
    f_csv.writerows(rows)
