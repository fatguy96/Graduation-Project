import csv
import datetime
import time


filename = "data/k1135_20_L34.csv"
csv_reader = csv.reader(open(filename))

has_handle = []

start_day = time.time()
time_temp = time.time()
time1 = time.time()
time2 = time.time()

i = 0
for row in csv_reader:

    time_temp = time.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    Y, m, d, H, M, S = time_temp[0:6]
    time_row = datetime.datetime(Y, m, d, H, M, S)

    if i == 0:
        start_day = time_row
        time1 = start_day
        has_handle.append(row)
    else:
        if i % 2 == 1:
            time2 = time_row
            interval = time2 - time1
            sample_num = int(interval.total_seconds()/300)
            if sample_num == 1:
                has_handle.append(row)
            else:
                for j in range(sample_num - 1):
                    if 288 * 7 > has_handle.__len__():
                        print('请手动更改前七天的数据')
                    else:
                        time1_tem = time1 + datetime.timedelta(minutes=(j + 1) * 5)
                        tem_string = (time1_tem, has_handle[-288*7][1], has_handle[-288*7][2])
                        has_handle.append(tem_string)
                has_handle.append(row)
        else:
            time1 = time_row
            interval = time1 - time2
            sample_num = int(interval.total_seconds() / 300)
            if sample_num == 1:
                has_handle.append(row)
            else:
                for j in range(sample_num - 1):
                    if 288 * 7 > has_handle.__len__():
                        print('请手动更改前七天的数据')
                    else:
                        time2_tem = time2 + datetime.timedelta(minutes=(j + 1) * 5)
                        tem_string = (time2_tem, has_handle[-288*7][1], has_handle[-288*7][2])
                        has_handle.append(tem_string)
                has_handle.append(row)
    i += 1

with open('k1135_20_L34_has.csv', 'w+') as f:
    f_csv = csv.writer(f, )
    f_csv.writerows(has_handle)
