from random import randint
from math import floor
import matplotlib.pyplot as plt
import numpy as np


class Population:
    def __init__(self, indivs=None):
        if not indivs:
            self.population = None
            self.pop_size = 0
        else:
            self.population = np.array(indivs)
            self.pop_size = self.population.shape[0]

    def get_by_idx(self, index):
        return self.population[index]

    def gen_rand_pop(self, pop_size, genes_num):
        self.pop_size = pop_size
        pop_temp = []
        for i in range(pop_size):
            pop_temp.append([randint(0, 9) // 9 for _ in range(genes_num)])  # 90% for 0, 10% for 1
        self.population = np.array(pop_temp)

    def calc_fitness(self, task):
        fitness_temp = []

        values = np.array([t.c for t in task.items])
        weights = np.array([t.w for t in task.items])
        sizes = np.array([t.s for t in task.items])

        sum_weights = self.population @ np.transpose(weights)
        sum_sizes = self.population @ np.transpose(sizes)
        sum_values = self.population @ np.transpose(values)

        for i in range(self.pop_size):
            if sum_sizes[i] > task.s or sum_weights[i] > task.w:
                fitness_temp.append([0])
            else:
                fitness_temp.append([sum_values[i]])

        self.fitness_arr = np.array(fitness_temp)

    def get_rand_pop_slice(self, slice_size):
        chosen_arr = np.random.choice([x for x in range(self.pop_size)], size=slice_size, replace=False)
        return chosen_arr, np.array([self.fitness_arr[idx] for idx in chosen_arr])

    def best(self):
        return self.population[self.fitness_arr.argmax()]


class Item:
    def __init__(self, w, s, c):
        self.w = w
        self.s = s
        self.c = c


class Task:
    def __init__(self, n, w, s):
        self.n = n
        self.w = w
        self.s = s
        self.items = []

    def push_item(self, item):
        self.items.append(item)


def generate_task(n, w, s, output_file):
    max_wi = floor(10 * w / n)
    max_si = floor(10 * s / n)
    w_sum = 0
    s_sum = 0

    while w_sum <= 2 * w or s_sum <= 2 * s:
        w_arr = []
        s_arr = []
        c_arr = []
        w_sum = 0
        s_sum = 0
        for i in range(n):
            wi = randint(1, max_wi)
            w_sum += wi
            w_arr.append(wi)

            si = randint(1, max_si)
            s_sum += si
            s_arr.append(si)

            c_arr.append(randint(1, n - 1))

    with open(output_file, 'w') as f:
        f.write(f'{n},{w},{s}\n')
        for i in range(n):
            f.write(f'{w_arr[i]},{s_arr[i]},{c_arr[i]}\n')


def read_task(input_file):
    try:
        with open(input_file, 'r') as f:
            n, w, s = f.readline().strip().split(',')
            task = Task(int(n), int(w), int(s))
            for line in f:
                w, s, c = line.strip().split(',')
                item = Item(int(w), int(s), int(c))
                task.push_item(item)
    except FileNotFoundError:
        print("\,!File not found!\n")
    else:
        return task


def mutate(genes, rate):
    genes_num = len(genes)
    to_mutate_num = floor(genes_num * rate)
    to_mutate = np.random.choice([x for x in range(genes_num)], size=to_mutate_num, replace=False)
    for gene_idx in to_mutate:
        genes[gene_idx] = int(not genes[gene_idx])
    return genes


def fitness(genes, task):
    score = 0
    w_sum = 0
    s_sum = 0
    for is_present, item in zip(list(genes), task.items):
        if is_present:
            score += item.c
            w_sum += item.w
            s_sum += item.s
            if w_sum > task.w or s_sum > task.s:
                score = 0
                break
    return score


def tournament(population, tourn_size):
    indexes, fitness_arr = population.get_rand_pop_slice(tourn_size)
    best_idx = indexes[np.argmax(fitness_arr)]
    return population.get_by_idx(best_idx)


def crossover(parent1, parent2, rate): # single point method
    if randint(0, 1) / 100 > rate:
        return parent1
    cutting_point = len(parent1) // 3
    return list(parent1[:cutting_point]) + list(parent2[cutting_point:])


if __name__ == '__main__':
    POP_SIZE = 700
    TOURN_SIZE = 600
    CROSS_RATE = 0.4
    MUT_RATE = 0.01
    ITERATIONS = 250
    scores_per_gen = []

    # generate_task(n=1001, w=10001, s=10001, output_file='task.csv')
    task = read_task(input_file='task.csv')
    pop = Population()
    pop.gen_rand_pop(POP_SIZE, task.n)
    i = 0
    while i < ITERATIONS:
        new_pop = []
        j = 0
        pop.calc_fitness(task)
        while j < POP_SIZE:
            parent1 = tournament(pop, TOURN_SIZE)
            parent2 = tournament(pop, TOURN_SIZE)
            child = crossover(parent1, parent2, CROSS_RATE)
            child = mutate(child, MUT_RATE)
            new_pop.append(child)
            j += 1
        best = pop.best()
        pop = Population(new_pop)
        scores_per_gen.append(fitness(best, task))
        print('Generation: ', i + 1, ', fitness: ', fitness(best, task), ', ones: ', sum(best), sep='')
        i += 1
    plt.plot([x for x in range(1, ITERATIONS + 1)], scores_per_gen)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
