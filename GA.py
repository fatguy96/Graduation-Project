import random
import tensorflow as tf
import numpy as np
from sklearn import preprocessing


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
            self.fitness.append(0)
            self.selector_probability.append(0)
            self.new_individuals.append(0)

    def ga_encoding(self, num):
        pop = [[]]
        for i in range(self.size):
            temp = []
            for j in range(num):
                temp.append(round(random.uniform(-0.5, 0.5), 8))
            pop.append(temp)
        return pop[1:]

    @staticmethod
    def fitness_func(individual):

        # 训练BP网络
        data = []
        with open('1135.csv', "r") as f:
            i = 0
            for line in f:
                sample = line.strip().split(',')
                if len(sample) == 15 and i >= 577:
                    data.append([float(sample[1]), float(sample[2]), float(sample[3]),
                                 float(sample[4]), float(sample[5]), float(sample[6]),
                                 float(sample[7]), float(sample[8]), float(sample[9]),
                                 float(sample[10]), float(sample[11]), float(sample[12]),
                                 float(sample[13]), float(sample[14])])
                i = i + 1
        data = np.array(data)
        column1 = data[:, 3:14]
        column2 = data[:, 0:3]
        x = np.mat(column1)
        y = np.mat(column2)

        # 正常归一化及还原，精度0.01
        scaler_x = preprocessing.MinMaxScaler()
        x_data = scaler_x.fit_transform(x)
        scaler_y = preprocessing.MinMaxScaler()
        y_data = scaler_y.fit_transform(y)

        tf.reset_default_graph()  # 重置默认图
        graph = tf.Graph()  # 新建空白图
        with graph.as_default() as g:  # 将新建的图作为默认图
            with tf.Session(graph=g) as sess:  # Session  在新建的图中运行

                xs = tf.placeholder(tf.float32, [None, 11])
                ys = tf.placeholder(tf.float32, [None, 3])

                w1 = []
                for i in range(11):
                    a = individual[9 * i:9 * i + 9]
                    w1.append(a)

                w2 = []
                for i in range(9):
                    b = individual[3 * i + 99:3 * (i + 1) + 99]
                    w2.append(b)

                b1 = individual[126:135]
                b2 = individual[135:138]

                Weights_1 = tf.Variable(tf.cast(np.mat(w1), dtype=tf.float32), name='Weights_1')
                biases_1 = tf.Variable(tf.cast(np.mat(b1), dtype=tf.float32), name='biases_1')
                Weights_2 = tf.Variable(tf.cast(np.mat(w2), dtype=tf.float32), name='Weights_2')
                biases_2 = tf.Variable(tf.cast(np.mat(b2), dtype=tf.float32), name='biases_2')

                z1 = tf.add(tf.matmul(xs, Weights_1), biases_1)
                a1 = tf.nn.relu(z1)
                y_ = tf.add(tf.matmul(a1, Weights_2), biases_2)

                loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_),
                                                    reduction_indices=[1]))
                train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

                sess.run(tf.global_variables_initializer())
                # 上面定义的都没有运算，直到 sess.run 才会开始运算
                _, error = sess.run([train_step, loss], feed_dict={xs: x_data, ys: y_data})
                error = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
                print("individuals fitness:%f" % (1 / error))
                return 1 / error

    # 计算个体的概率
    def evaluate(self):
        sp = self.selector_probability
        for i in range(self.size):
            # 个体的表现
            self.fitness[i] = self.fitness_func(self.individuals[i])
        # 总的表现
        ft_sum = sum(self.fitness)
        for i in range(self.size):
            sp[i] = self.fitness[i] / float(ft_sum)
        for i in range(1, self.size):
            # 累计概率
            sp[i] = sp[i] + sp[i - 1]

    # 进行选择
    def select(self):
        # 论转法
        (t, i) = (random.random(), 0)
        for p in self.selector_probability:
            if p > t:
                break
            i = i + 1
        return i

    # 进行交叉
    def cross(self, individual_1, individual_2):

        if random.random() < self.crossover_probability:
            a1 = random.random()
            a2 = random.random()
            temp1 = []
            for index in range(len(individual_1)):
                a = individual_1[index] * a1 + individual_2[index] * (1-a1)
                temp1.append(a)
            temp2 = []
            for index in range(len(individual_2)):
                a = individual_1[index] * a2 + individual_2[index] * (1-a2)
                temp2.append(a)
            individual_1, individual_2 = temp1, temp2
        return [individual_1, individual_2]

    # 进行变异
    def mutate(self, individual):

        py = len(individual)

        if random.random() < self.mutation_probability:
            index = random.randint(0, py - 1)
            individual[index] = round(random.uniform(-0.5, 0.5), 8)

        return individual

    def reproduct_elitist(self):
        # 与当前种群进行适应度比较，更新最佳个体
        j = -1
        for i in range(self.size):
            if self.elitist['fitness'] < self.fitness[i]:
                j = i
                self.elitist['fitness'] = self.fitness[i]
        if j >= 0:
            self.elitist['chromosome'] = self.individuals[j]
            self.elitist['age'] = self.age

        # 用当前最佳个体替换种群新个体中最差者
        new_fitness = [self.fitness_func(v) for v in self.new_individuals]
        best_fitness = max(new_fitness)
        if self.elitist['fitness'] > best_fitness:
            # 寻找最小适应度对应个体
            j = 0
            for i in range(self.size):
                if best_fitness > new_fitness[i]:
                    j = i
                    best_fitness = new_fitness[i]
            # 最佳个体取代最差个体
            self.new_individuals[j] = self.elitist['chromosome']

    def evolve(self):
        indvs = self.individuals
        new_indvs = self.new_individuals

        # 计算适应度及选择概率
        self.evaluate()

        # 进化操作
        i = 0
        while True:
            # 选择两名个体，进行交叉与变异，产生 2 名新个体
            idv1 = self.select()
            idv2 = self.select()

            # 交叉
            idv1_new = indvs[idv1]
            idv2_new = indvs[idv2]
            (idv1_new, idv2_new) = self.cross(idv1_new, idv2_new)

            # 变异
            idv1_new = self.mutate(idv1_new)
            idv2_new = self.mutate(idv2_new)

            if random.randint(0, 1) == 0:
                new_indvs[i] = idv1_new
            else:
                new_indvs[i] = idv2_new

            # 判断进化过程是否结束
            i = i + 1
            if i >= self.size:
                break

        # 最佳个体保留
        self.reproduct_elitist()

        # 更新换代
        for i in range(self.size):
            self.individuals[i] = self.new_individuals[i]

    def run(self):
        for i in range(self.generation_max):
            self.evolve()
            print(i, max(self.fitness), sum(self.fitness) / self.size, min(self.fitness))





