import csv
import os
import time

print(os.getcwd())
csv_reader = csv.reader(open('weather/K1135_20_weather1.csv'))

rows = []
for row in csv_reader:
    csv_reader1 = csv.reader(open('K1135+20_all.csv'))
    for row1 in csv_reader1:
        temp = time.strptime(row1[0], '%Y-%m-%d %H:%M:%S')
        Y, m, d, H, M, S = temp[0:6]
        if d < 10:
            temp_row1 = '{}年0{}月0{}日'.format(Y, m, d)
        else:
            temp_row1 = '{}年0{}月{}日'.format(Y, m, d)
        if row[0] == temp_row1:
            row1.append(row[1])
            row1.append(row[2])
            rows.append(row1)


with open('sample/K1135_20_allflow_version3.csv', 'w+') as f:
    f_csv = csv.writer(f,)
    f_csv.writerows(rows)



