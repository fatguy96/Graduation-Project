# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import matplotlib
print(matplotlib.matplotlib_fname())


def to_percent(y, position):
    return str(100 * y) + '%'


plt.figure()
plt.title("Relative Error")
names = ['SVR', 'GA-BP', 'LSTM', 'MIX']
x = range(len(names))
y = [0.1059, 0.1070, 0.1078, 0.1064]
y1 = [0.1045, 0.1082, 0.1091, 0.1072]
for a, b in zip(x, y):
    plt.annotate("{}%".format(round(100*b, 2)), xy=(a, b))

for a, b in zip(x, y1):
    plt.annotate("{}%".format(round(100*b, 2)), xy=(a, b))

plt.plot(x, y1, 'bo-', label='single')
plt.plot(x, y, 'ko-', label='multi-source')
plt.legend(loc='upper right')

plt.xticks(x, names, rotation=45)
plt.margins(0.08)
plt.subplots_adjust(bottom=0.15)

formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)

plt.legend(loc='upper right')
plt.show()