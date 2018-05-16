import numpy as np
from My_SVR.svr import My_SVR


def load_data(filename='data/data.csv'):
    data = []
    i = 1
    with open(filename, 'r') as f:
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
    return data


data = load_data()

svr = My_SVR(data, cv=6)

svr.verify()
current = [[145, 132, 139, 134, 102, 108, 144, 116, 137, 55, 33]]
data_ = np.array(current)
print(svr.predict(data_))