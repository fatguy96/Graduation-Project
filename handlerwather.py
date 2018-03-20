import csv
import os
import time

print(os.getcwd())
csv_reader = csv.reader(open('K1135_20_weather.csv'))

rows = []
for row in csv_reader:
    csv_reader1 = csv.reader(open('../sample/K1135_20_L31_version1.csv'))
    for row1 in csv_reader1:
        temp = time.strptime(row1[0], '%Y-%m-%d %H:%M:%S')
        Y, m, d, H, M, S = temp[0:6]
        if d < 10:
            row1[0] = '{}年0{}月0{}日'.format(Y, m, d)
        else:
            row1[0] = '{}年0{}月{}日'.format(Y, m, d)
        if row[0] == row1[0]:
            content = (row1, row[1], row[3])
            rows.append(content)


with open('sample/K1135_20_L31_version2.csv', 'a') as f:
    f_csv = csv.writer(f,)
    f_csv.writerows(rows)



