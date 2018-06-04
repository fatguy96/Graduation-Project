import random

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class GA:

    def __init__(self, size, gen_max, num, cp, mp, sample, label, sample_v, label_v):

        # 定义神经网络
        # -----------------------------------------
        self.xs = tf.placeholder(tf.float32, [None, 11])
        self.ys = tf.placeholder(tf.float32, [None, 3])

        self.Weights_1 = tf.Variable(tf.random_normal([11, 9], stddev=1, seed=1, mean=0))
        self.biases_1 = tf.Variable(tf.random_normal([1, 9], stddev=1, seed=1, mean=0))
        self.Weights_2 = tf.Variable(tf.random_normal([9, 3], stddev=1, seed=1, mean=0))
        self.biases_2 = tf.Variable(tf.random_normal([1, 3], stddev=1, seed=1, mean=0))

        self.z1 = tf.add(tf.matmul(self.xs, self.Weights_1), self.biases_1)
        self.a1 = tf.nn.relu(self.z1)
        self.y_ = tf.add(tf.matmul(self.a1, self.Weights_2), self.biases_2)

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.y_),
                                                 reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(0.05).minimize(self.loss)

        # -----------------------------------------

        self.individuals = []  # 个体集/种群
        self.fitness = []  # 表现
        self.selector_probability = []  # 选择的概率
        self.new_individuals = []  # 新的个体
        self.elitist = {'chromosome': None, 'fitness': 0, 'age': 0}
        self.size = size  # 种群的大小
        self.crossover_probability = cp  # 交叉的概率
        self.mutation_probability = mp  # 变异的概率
        self.generation_max = gen_max  # 更新的代数
        self.age = 0  # 当前的代数
        self.sample = sample
        self.label = label
        self.sample_verify = sample_v
        self.label_verify = label_v

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

    def fitness_func(self, individual):

        x = np.mat(self.sample)
        y = np.mat(self.label)

        # 正常归一化及还原，精度0.01
        scaler_x = preprocessing.MinMaxScaler()
        x_data = scaler_x.fit_transform(x)
        scaler_y = preprocessing.MinMaxScaler()
        y_data = scaler_y.fit_transform(y)

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
        update1 = tf.assign(self.Weights_1, tf.cast(np.mat(w1), dtype=tf.float32))
        update2 = tf.assign(self.biases_1, tf.cast(np.mat(b1), dtype=tf.float32))
        update3 = tf.assign(self.Weights_2, tf.cast(np.mat(w2), dtype=tf.float32))
        update4 = tf.assign(self.biases_2, tf.cast(np.mat(b2), dtype=tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([update1, update2, update3, update4])
            # 上面定义的都没有运算，直到 sess.run 才会开始运算
            _, error = sess.run([self.train_step, self.loss], feed_dict={self.xs: x_data, self.ys: y_data})
            error = sess.run(self.loss, feed_dict={self.xs: x_data, self.ys: y_data})
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

    def train_net(self, individual):

        x = np.mat(self.sample)
        y = np.mat(self.label)

        # 正常归一化及还原，精度0.01
        scaler_x = preprocessing.MinMaxScaler()
        x_data = scaler_x.fit_transform(x)
        scaler_y = preprocessing.MinMaxScaler()
        y_data = scaler_y.fit_transform(y)

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

        update1 = tf.assign(self.Weights_1, tf.cast(np.mat(w1), dtype=tf.float32))
        update2 = tf.assign(self.biases_1, tf.cast(np.mat(b1), dtype=tf.float32))
        update3 = tf.assign(self.Weights_2, tf.cast(np.mat(w2), dtype=tf.float32))
        update4 = tf.assign(self.biases_2, tf.cast(np.mat(b2), dtype=tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([update1, update2, update3, update4])
            # 上面定义的都没有运算，直到 sess.run 才会开始运算
            plt.xlabel("Train time")
            plt.ylabel("Mean Squared Error")
            plt.title('Learning curves')
            for i in range(2000):
                _, loss_ = sess.run([self.train_step, self.loss], feed_dict={self.xs: x_data, self.ys: y_data})
                plt.plot(i, loss_, 'r.', label='loss')
                if i % 100 == 0:
                    print('epochs{}: {}'.format(i, loss_))
            plt.legend(loc="best")
            plt.savefig('ga_loss/loss.png')
            plt.close()
            saver.save(sess, 'ga_model/ga')

    def verify(self):
        x = np.mat(self.sample_verify)
        y = np.mat(self.label_verify)

        # 正常归一化及还原，精度0.01
        scaler_x = preprocessing.MinMaxScaler()
        x_data = scaler_x.fit_transform(x)
        scaler_y = preprocessing.MinMaxScaler()
        y_data = scaler_y.fit_transform(y)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ga_model/')
            saver.restore(sess, model_file)
            prediction_value = sess.run(self.y_, feed_dict={self.xs: x_data})

            real_pre = scaler_y.inverse_transform(prediction_value)

            model_names = ["GA_bp"]
            model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error,
                                  r2_score]  # 回归评估指标对象集
            acc = np.average(np.abs(real_pre[:, 0:1] - y[:, 0:1])/y[:, 0:1])  # 偏差
            model_metrics_list = []  # 回归评估指标列表

            tmp_list = []  # 每个内循环的临时结果列表
            for m in model_metrics_name:  # 循环每个指标对象
                tmp_score = m(prediction_value[:, 0:1], y_data[:, 0:1])  # 计算每个回归指标结果
                tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
            tmp_list.append("{} %".format(round(acc*100, 2)))
            model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

            df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2', 'acc'])  # 建立回归指标的数据框

            print('regression metrics:')  # 打印输出标题
            print(df2)  # 打印输出回归指标的数据框
            print(70 * '-')  # 打印分隔线
            print('short name \t full name')  # 打印输出缩写和全名标题
            print(' ev \t\t\t explained_variance')
            print('mae \t\t\t mean_absolute_error')
            print('mse \t\t\t mean_squared_error')
            print(' r2 \t\t\t r2')
            print('acc \t\t\t 相对误差')
            print(70 * '-')  # 打印分隔线
            result = y[:, 0:1] - real_pre[:, 0:1]
            result = result.reshape(-1, 1)
            re = []
            re_sum = 0
            for i in range(0, len(result)):
                re_sum = re_sum + abs(round(float(result[i]), 8))
                re.append(round(float(result[i]), 8))

            epoch = int(len(result) / 288)
            for i in range(epoch):
                plt.figure()
                plt.xlabel("time")
                plt.ylabel("number")
                plt.plot(real_pre[i * 288: (i + 1) * 288, 0:1], 'r', label="Predictive value")
                plt.plot(y[i * 288: (i + 1) * 288, 0:1], 'b', label="Actual value")
                plt.legend(loc='upper right')
                plt.savefig('ga_test/test%d.png' % (i + 1))
                plt.close()

    def predict(self, predict_x):
        x = np.mat(self.sample)
        y = np.mat(self.label)

        # 正常归一化及还原，精度0.01
        scaler_x = preprocessing.MinMaxScaler()
        scaler_x.fit_transform(x)
        predict_x = scaler_x.transform(predict_x)
        scaler_y = preprocessing.MinMaxScaler()
        scaler_y.fit_transform(y)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ga_model/')
            saver.restore(sess, model_file)
            prediction_value = sess.run(self.y_, feed_dict={self.xs: predict_x})

            real_pre = scaler_y.inverse_transform(prediction_value)
            return real_pre

    def run(self):
        for i in range(self.generation_max):
            self.evolve()
            print(i, max(self.fitness), sum(self.fitness) / self.size, min(self.fitness))
        self.train_net(self.elitist['chromosome'])






