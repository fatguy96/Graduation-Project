import random
import tensorflow as tf
import numpy as np
import csv
from sklearn import preprocessing


# 添加层
def add_layer(inputs, Weights, biases, activation_function=None):
    # add one more layer and return the output of this layer
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


class GA:
    def __init__(self, size, gen_max, num, cp, mp):
        self.individuals = []  # 个体集/种群
        self.fitness = []  # 表现
        self.selector_probability = []  # 选择的概率
        self.new_individuals = []  # 新的个体
        self.last_gen = []  # 最后的选择
        self.elitist = {'fitness': 0, 'age': 0}
        self.size = size  # 种群的大小
        self.crossover_probability = cp  # 交叉的概率
        self.mutation_probability = mp  # 变异的概率
        self.generation_max = gen_max  # 更新的代数
        self.age = 0  # 当前的代数

        self.individuals = self.ga_encoding(num)  # 初始化种群中的个体
        for i in range(size):
            self.fitness[i] = 0
            self.selector_probability[i] = 0
            self.new_individuals[i] = 0

    def ga_encoding(self, num):
        pop = [[]]
        for i in range(self.size):
            temp = []
            for j in range(num):
                temp.append(round(random.uniform(-0.5, 0.5), 8))
            pop.append(temp)
        return pop[1:]

    def fitness_func(self, individual):
        pass



