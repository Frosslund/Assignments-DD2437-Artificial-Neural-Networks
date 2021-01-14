import numpy as np
from matplotlib import pyplot as plt
import itertools
import random


class HopfieldNet():

    def __init__(self, patterns, asynch=True):
        self.patterns = patterns
        self.weights = self.initialize_weights()
        self.asynch = asynch

    def initialize_weights(self):
        w_tot = []
        for p in range(self.patterns.shape[0]):
            w_curr = np.matmul(self.patterns[p].T, self.patterns[p])
            w_curr = np.subtract(w_curr, np.identity(
                w_curr.shape[0], dtype=int))
            w_tot.append(w_curr)

        w_sum = np.empty(shape=w_curr.shape)
        for i in range(len(w_tot)):
            if i == 0:
                w_sum = w_tot[i]
            else:
                w_sum = np.add(w_tot[i], w_sum)
        return w_sum

    def recall(self, pattern, rand=False):
        if self.asynch:
            print("tjatja")
            pre_pattern = pattern
            stable_count = 0
            iterations = 10
            order = np.arange(pattern.shape[1])
            if rand:
                random.shuffle(order)
            while (iterations > 0):
                for i in order:
                    curr = self.weights[:, [i]]
                    result = np.sign(np.dot(curr.T, pre_pattern.T))
                    if pre_pattern[:, i] != result:
                        pre_pattern[:, i] = result
                        stable_count = 0
                    stable_count += 1
                if stable_count >= 2*1024:
                    return pre_pattern
                iterations -= 1

        else:
            pre_pattern = pattern
            for i in range(9):
                after_pattern = np.sign(np.matmul(pre_pattern, self.weights))
                if ((pre_pattern == after_pattern).all()):
                    return after_pattern
                else:
                    pre_pattern = after_pattern
            return pre_pattern
        return pre_pattern

    def attractors(self, all_patterns):
        attractors = []
        for pattern in all_patterns:
            pre_pattern = pattern
            for i in range(30):
                after_pattern = np.sign(np.matmul(pre_pattern, self.weights))
                if ((pre_pattern == after_pattern).all()):
                    attractors.append(after_pattern)
                else:
                    pre_pattern = after_pattern
        return attractors

    def calc_energy(self, pattern):
        energy_sum = 0
        n = self.weights.shape[0]
        for i in range(n):
            for j in range(n):
                energy_sum += self.weights[i, j]*pattern[:, i]*pattern[:, j]
        return -energy_sum

    def calc_energy_fast(self, pattern):
        outer = np.outer(pattern, np.transpose(pattern))
        energy_sum = np.sum(np.multiply(self.weights, outer))
        return np.multiply(-1, energy_sum)

    def calc_energy_3(self, pattern):
        return -np.dot(pattern, self.weights).dot(pattern.T)


def task3_1():
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1]).reshape(1, -1)

    x = np.array([x1, x2, x3])

    hopfield = HopfieldNet(x)

    # Assert that the net has stored the three patterns
    """ np.testing.assert_array_equal(x1, hopfield.batch_test(x1))
    np.testing.assert_array_equal(x2, hopfield.batch_test(x2))
    np.testing.assert_array_equal(x3, hopfield.batch_test(x3)) """

    x1_d = np.array([1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
    x2_d = np.array([1, 1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x3_d = np.array([1, 1, 1, -1, 1, 1, -1, 1]).reshape(1, -1)

    """ np.testing.assert_array_equal(x1, hopfield.batch_test(x1_d))
    np.testing.assert_array_equal(x2, hopfield.batch_test(x2_d))
    np.testing.assert_array_equal(x3, hopfield.batch_test(x3_d)) """

    """ print(hopfield.batch_test(x1_d))
    print(hopfield.batch_test(x2_d)) """
    # print(hopfield.recall(x3_d))

    """ all_patterns = []
    lst = list(itertools.product([-1, 1], repeat=8))

    for i in range(len(lst)):
        all_patterns.append(list(lst[i]))

    #all_patterns = all_patterns.astype(int)

    # print(len(all_patterns))

    attract = hopfield.attractors(all_patterns)
    unique = np.unique(attract, axis=0)
    print(unique)
    print(unique.shape[0]) """
    x1_daf = np.array([-1, 1, -1, -1, 1, 1, -1, 1]).reshape(1, -1)
    x2_daf = np.array([1, -1, 1, -1, 1, 1, 1, -1]).reshape(1, -1)
    x3_daf = np.array([-1, 1, -1, 1, 1, -1, 1, 1]).reshape(1, -1)

    print(hopfield.recall(x3_daf))


def task3_2():
    data = np.loadtxt('annda_lab3/pict.dat', dtype=int,
                      delimiter=',').reshape(-1, 1024)

    p1 = np.array(data[0]).reshape(1, -1)
    p2 = np.array(data[1]).reshape(1, -1)
    p3 = np.array(data[2]).reshape(1, -1)
    p10 = np.array(data[9]).reshape(1, -1)
    p11 = np.array(np.copy(data[10])).reshape(1, -1)
    p11_copy = np.copy(p11)

    print('148')
    plt.imshow(p11.reshape((32, 32)), cmap="gray")
    plt.show()

    x = np.array([p1, p2, p3])
    hopfield = HopfieldNet(x)

    """ recall_1 = hopfield.recall(p1)
    recall_2 = hopfield.recall(p2)
    recall_3 = hopfield.recall(p3) """
    #recall_10 = hopfield.recall(p10)
    #recall_10 = hopfield.recall(p10, rand=True)
    recall_11 = hopfield.recall(p11_copy)

    print('162')
    plt.imshow(p11.reshape((32, 32)), cmap="gray")
    plt.show()


def task3_3():
    data = np.loadtxt('pict.dat', dtype=int,
                      delimiter=',').reshape(-1, 1024)

    p1 = np.array(data[0]).reshape(1, -1)
    p2 = np.array(data[1]).reshape(1, -1)
    p3 = np.array(data[2]).reshape(1, -1)
    p10 = np.array(data[9]).reshape(1, -1)
    p11 = np.array(data[10]).reshape(1, -1)

    x = np.array([p1, p2, p3])

    hopfield = HopfieldNet(x)

    """ plt.imshow(p10.reshape((32, 32)), cmap="gray")
    plt.show()
    plt.imshow(p11.reshape((32, 32)), cmap="gray")
    plt.show() """

    p1_e = hopfield.calc_energy_fast(p1)
    p2_e = hopfield.calc_energy_fast(p2)
    p3_e = hopfield.calc_energy_fast(p3)
    p10_e = hopfield.calc_energy_fast(p10)
    p11_e = hopfield.calc_energy_fast(p11)

    print(p1_e)
    print(p2_e)
    print(p3_e)
    print(p10_e)
    print(p11_e)


if __name__ == "__main__":
    task3_2()
