import csv
import datetime
import time

# 检查文件中每条记录是否为存在缺失
# 即判断每天记录是否间隔5分钟
csv_reader = csv.reader(open('data/k1135_20_L32.csv'))
i = 0
start_day = time.time()
time1 = time.time()
time2 = time.time()
for row in csv_reader:
    i += 1
    if i == 1:
        temp = time.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        Y, m, d, H, M, S = temp[0:6]
        start_day = datetime.datetime(Y, m, d, H, M, S)
    if i % 2 == 1:
        first = time.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        Y, m, d, H, M, S = first[0:6]
        time1 = datetime.datetime(Y, m, d, H, M, S)
        if i != 1:
            interval = time1 - time2
            if interval.seconds != 300 or interval.days != 0:
                print(time2, "和", time1, "间隔不为5分钟")
                # print("丢失记录数：%d" % (interval.total_seconds()/300))
    if i % 2 == 0:
        second = time.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        Y, m, d, H, M, S = second[0:6]
        time2 = datetime.datetime(Y, m, d, H, M, S)
        interval = time2 - time1
        if interval.seconds != 300 or interval.days != 0:
            print(time1, "和", time2, "间隔不为5分钟")
            # print("丢失记录数：%d" % (interval.total_seconds() / 300))

print()
if i % 2 == 0:
    print(start_day, "-->", time2)
    total_interval = time2 - start_day
else:
    print(start_day, "-->", time2)
    total_interval = time1 - start_day
print("理论记录一共%d条" % ((total_interval.total_seconds()/300) + 1))
print("实际记录一共%d条" % i)

