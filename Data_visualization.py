import csv
import os
import matplotlib.pyplot as plt
print(os.getcwd())
csv_reader = csv.reader(open('sample/K1135_20_L31_version1.csv'))
rows = []
for row in csv_reader:
    rows.append(int(row[1]))

epoch = int(rows.__len__()/288)

print(max(rows))
print(rows.index(120))
for i in range(epoch):
    plt.figure()
    plt.xlabel("time")
    plt.ylabel("number")
    plt.plot(rows[i*288:(i+1)*288])
    plt.savefig('/home/fate/Desktop/Traffic-flow/day%d.png' % (i+1))
    plt.close()
