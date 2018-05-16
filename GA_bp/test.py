import numpy as np

from GA_bp.GA import GA

data = []
with open('data/data.csv', "r") as f:
    i = 0
    for line in f:
        sample = line.strip().split(',')
        if len(sample) == 15 and i >= 577:
            data.append([float(sample[1]), float(sample[2]), float(sample[3]),
                         float(sample[4]), float(sample[5]), float(sample[6]),
                         float(sample[7]), float(sample[8]), float(sample[9]),
                         float(sample[10]), float(sample[11]), float(sample[12]),
                         float(sample[13]), float(sample[14])])
        i += 1
data = np.array(data)
train_size = int(0.7 * len(data))
column1 = data[:train_size, 3:14]
column2 = data[:train_size, 0:3]
column1_v = data[train_size:, 3:14]
column2_v = data[train_size:, 0:3]
a = GA(30, 50, 138, 0.75, 0.05, column1, column2, column1_v, column2_v)
a.run()