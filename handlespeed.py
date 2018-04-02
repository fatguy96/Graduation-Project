import csv
import numpy as np
import matplotlib.pyplot as plt
csv_reader = csv.reader(open('data/data_org.csv'))

contents = []
for row in csv_reader:
    if row[1] == "31" and row[23] == "K1135+20":
        content = [row[0], row[3], row[5], row[8], row[10], row[12], row[14], row[16]]
        contents.append(content)

with open("L31_speed.csv", 'w+') as f:
    f_csv = csv.writer(f, )
    f_csv.writerows(contents)

contents = np.array(contents)
print(contents.shape)
image_number = int(contents.__len__()/288)
for i in range(image_number):
    plt.figure()
    plt.title("L31_speed")
    plt.xlabel("time/5min")
    plt.ylabel("km/h")
    # plt.plot(speed1[i * 288:(i + 1) * 288], 'k.', label="1")
    plt.plot((contents[i * 288:(i + 1) * 288, 1]).reshape(-1, 1), 'r.', label="1")
    # plt.plot(speed3[i * 288:(i + 1) * 288], 'indianred', label="3")
    # plt.plot(speed4[i * 288:(i + 1) * 288], 'r', label="4")
    # plt.plot(speed5[i * 288:(i + 1) * 288], 'y', label="5")
    # plt.plot(speed6[i * 288:(i + 1) * 288], 'peru', label="6")
    # plt.plot(speed7[i * 288:(i + 1) * 288], 'palegreen', label="7")
    plt.legend()
    plt.savefig('/home/fate/Desktop/Traffic-speed/day%d.png' % (i+1))
    plt.close()

