from random import randint
from math import floor
import matplotlib.pyplot as plt
import numpy as np

N_GLOB = 1001
W_GLOB = 10001
S_GLOB = 10001


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

    def generate_rand_pop(self, pop_size, genes_num):
        self.pop_size = pop_size
        pop_temp = []
        for i in range(pop_size):
            pop_temp.append([randint(0, 4) // 4 for _ in range(genes_num)])
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

    def best(self):
        return self.population[self.fitness_arr.argmax()], self.fitness_arr.max(initial=0)


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
    with open(input_file, 'r') as f:
        n, w, s = f.readline().split(',')
        task = Task(int(n), int(w), int(s))
        for line in f:
            w, s, c = line.split(',')
            item = Item(int(w), int(s), int(c))
            task.push_item(item)
    return task


def mutate(genes, rate):
    genes_num = len(genes)
    to_mutate_num = floor(genes_num * rate)
    to_mutate = np.random.randint(genes_num, size=to_mutate_num)
    for gene_idx in to_mutate:
        genes[gene_idx] = int(not genes[gene_idx])


def tournament(population, tourn_size):
    indexes = np.random.randint(population.pop_size, size=tourn_size)
    fitness_arr = list(map(lambda idx: population.fitness_arr[idx], indexes))
    best_idx = indexes[np.argmax(fitness_arr)]
    return population.get_by_idx(best_idx)


def crossover(parent1, parent2, probab):
    if randint(0, 1) / 100 > probab:
        return parent1
    cutting_point = len(parent1) // 3
    return list(parent1[:cutting_point]) + list(parent2[cutting_point:])


def knapsack(task, POP_SIZE=1000, TOURN_SIZE=200, CROSS_RATE=0.5, MUT_RATE=0.001, ITERATIONS=250):
    scores_per_gen = []

    pop = Population()
    pop.generate_rand_pop(POP_SIZE, task.n)
    i = 1
    while i < ITERATIONS:
        pop.calc_fitness(task)
        _, best_fit = pop.best()
        scores_per_gen.append(best_fit)
        print('Generation:', i, ', fitness:', best_fit)

        new_pop = []
        j = 0
        while j < POP_SIZE:
            parent1 = tournament(pop, TOURN_SIZE)
            parent2 = tournament(pop, TOURN_SIZE)
            child = crossover(parent1, parent2, CROSS_RATE)
            mutate(child, MUT_RATE)
            new_pop.append(child)
            j += 1
        pop = Population(new_pop)
        i += 1

    pop.calc_fitness(task)
    best, best_fit = pop.best()
    scores_per_gen.append(best_fit)
    print('Generation:', i, ', fitness:', best_fit)
    return best, np.array(scores_per_gen)


def greedySearch(task):
    items = [(item.c / (item.w + item.s), item.c, item.w, item.s) for item in task.items]
    values_sorted = sorted(items, key=lambda x: x[0], reverse=True)
    w_sum = c_sum = s_sum = 0
    for item in values_sorted:
        if w_sum > task.w or s_sum > task.s:
            break
        c_sum += item[1]
        w_sum += item[2]
        s_sum += item[3]
    return c_sum


def evol_vs_nonevol(tests_num, nonevol, **evol_kwargs):
    nonevol_fit = 0
    evol_fit = 0
    for i in range(tests_num):
        generate_task(N_GLOB, W_GLOB, S_GLOB, 'task.csv')
        task = read_task('task.csv')
        nonevol_fit += nonevol(task) // tests_num
        evol_fit += knapsack(task, **evol_kwargs)[1][-1] // tests_num

    print("Greedy fitness:", nonevol_fit)
    print("Genetic fitness:", evol_fit)
    plt.bar(['Greedy Algorithm', 'Genetic Algorithm'], [nonevol_fit, evol_fit])
    plt.ylabel("Fitness")
    # plt.show()
    plt.savefig('evol_vs_nonevol.png')


def crossoverTest(tests_num, cross_rates, iterations, **kwargs):
    print("Analysis of the crossover probability:")
    fit_arr = [0 for _ in range(len(cross_rates))]
    for test_num in range(tests_num):
        generate_task(N_GLOB, W_GLOB, S_GLOB, 'task.csv')
        task = read_task('task.csv')
        i = 0
        for cross_rate in cross_rates:
            _, fitness = knapsack(task, CROSS_RATE=cross_rate, ITERATIONS=iterations, **kwargs)
            fit_arr[i] += fitness // tests_num
            if test_num == tests_num - 1:
                plt.plot([x for x in range(1, iterations + 1)], fit_arr[i], label=str(cross_rates[i]))
            i += 1

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    # plt.show()
    plt.savefig('crossover.png')


def mutationTest(tests_num, mut_rates, iterations, **kwargs):
    print("Analysis of the mutation probability:")
    fit_arr = [0 for _ in range(len(mut_rates))]
    for test_num in range(tests_num):
        generate_task(N_GLOB, W_GLOB, S_GLOB, 'task.csv')
        task = read_task('task.csv')
        i = 0
        for mut_rate in mut_rates:
            _, fitness = knapsack(task, MUT_RATE=mut_rate, ITERATIONS=iterations, **kwargs)
            fit_arr[i] += fitness // tests_num
            if test_num == tests_num - 1:
                plt.plot([x for x in range(1, iterations + 1)], fit_arr[i], label=str(mut_rates[i]))
            i += 1

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    # plt.show()
    plt.savefig('mutation.png')


def tournamentTest(tests_num, tourn_sizes, iterations, **kwargs):
    print("Analysis of the tournament size:")
    fit_arr = [0 for _ in range(len(tourn_sizes))]
    for test_num in range(tests_num):
        generate_task(N_GLOB, W_GLOB, S_GLOB, 'task.csv')
        task = read_task('task.csv')
        i = 0
        for tourn_size in tourn_sizes:
            _, fitness = knapsack(task, TOURN_SIZE=tourn_size, ITERATIONS=iterations, **kwargs)
            fit_arr[i] += fitness // tests_num
            if test_num == tests_num - 1:
                plt.plot([x for x in range(1, iterations + 1)], fit_arr[i], label=str(tourn_sizes[i]))
            i += 1

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    # plt.show()
    plt.savefig('tournament.png')


def populationTest(tests_num, pop_sizes, iterations, **kwargs):
    print("Analysis of the population size:")
    fit_arr = [0 for _ in range(len(pop_sizes))]
    for test_num in range(tests_num):
        generate_task(N_GLOB, W_GLOB, S_GLOB, 'task.csv')
        task = read_task('task.csv')
        i = 0
        for pop_size in pop_sizes:
            _, fitness = knapsack(task, POP_SIZE=pop_size, ITERATIONS=iterations, **kwargs)
            fit_arr[i] += fitness // tests_num
            if test_num == tests_num - 1:
                plt.plot([x for x in range(1, iterations + 1)], fit_arr[i], label=str(pop_sizes[i]))
            i += 1

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    # plt.show()
    plt.savefig('population.png')


def best_vs_worst(tests_num, best, worst, iterations):
    print("Comparing the best with the worst parameters:")
    fit_arr = [0, 0]
    for test_num in range(tests_num):
        generate_task(N_GLOB, W_GLOB, S_GLOB, 'task.csv')
        task = read_task('task.csv')
        _, fitness_best = knapsack(task, *best, ITERATIONS=iterations)
        _, fitness_worst = knapsack(task, *worst, ITERATIONS=iterations)
        fit_arr[0] += fitness_best // tests_num
        fit_arr[1] += fitness_worst // tests_num

    plt.plot([x for x in range(1, iterations + 1)], fit_arr[0], label='The best')
    plt.plot([x for x in range(1, iterations + 1)], fit_arr[1], label='The worst')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    # plt.show()
    plt.savefig('best_vs_worst.png')


if __name__ == '__main__':
    crossoverTest(tests_num=5, cross_rates=[0.9, 0.5, 0.1], POP_SIZE=100, iterations=600)
    print('Crossover test finished')
    mutationTest(tests_num=5, mut_rates=[0.01, 0.005, 0.001], iterations=600)
    print('Mutation test finished')
    tournamentTest(tests_num=5, tourn_sizes=[100, 500, 900], iterations=600)
    print('Tournament test finished')
    populationTest(tests_num=5, pop_sizes=[100, 1000, 3000], iterations=1000)
    print('Population test finished')
    evol_vs_nonevol(5, greedySearch, POP_SIZE=100, TOURN_SIZE=10, CROSS_RATE=0.9, MUT_RATE=0.001, ITERATIONS=800)
    print('nonevol vs evol comparison finished')
    thebest = [3000, 100, 0.9, 0.001]
    theworst = [100, 90, 0.1, 0.01]
    best_vs_worst(5, thebest, theworst, iterations=1000)
    print('the best vs the worst comparison finished')
