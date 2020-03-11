from random import randint
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

# def matmul(args):
#     pop = args[0]
#     mat = args[1]
#     return pop @ np.transpose(mat)


class Population:
    def __init__(self, indivs=None):
        if not indivs:
            self.population = None
            self.pop_size = 0
        else:
            self.population = np.array(indivs)
            self.pop_size = self.population.shape[0]
            self.indexesArr = [ind for ind in range(len(indivs))]

    def get_by_idx(self, index):
        return self.population[index]

    def generate_rand_pop(self, pop_size, genes_num):
        self.pop_size = pop_size
        pop_temp = []
        for i in range(pop_size):
            pop_temp.append([randint(0, 4) // 4 for _ in range(genes_num)])
        self.population = np.array(pop_temp)
        self.indexesArr = [ind for ind in range(pop_size)]


    def calc_fitness(self, task):
        fitness_temp = []

        values = np.array([t.c for t in task.items])
        weights = np.array([t.w for t in task.items])
        sizes = np.array([t.s for t in task.items])

        sum_weights = self.population @ np.transpose(weights)
        sum_sizes = self.population @ np.transpose(sizes)
        sum_values = self.population @ np.transpose(values)
        # pop = self.population
        # p = mp.Pool(5)
        # sum_weights, sum_sizes, sum_values = p.map(matmul, [(pop,weights), (pop,sizes), (pop,values)])

        for i in range(self.pop_size):
            if sum_sizes[i] > task.s or sum_weights[i] > task.w:
                fitness_temp.append([0])
            else:
                fitness_temp.append([sum_values[i]])

        self.fitness_arr = np.array(fitness_temp)

    def get_rand_pop_slice(self, slice_size):
        chosen_arr = np.random.randint(self.pop_size, size=slice_size)
        return chosen_arr, list(map(lambda idx: self.fitness_arr[idx], chosen_arr))

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
    try:
        with open(input_file, 'r') as f:
            n, w, s = f.readline().strip().split(',')
            task = Task(int(n), int(w), int(s))
            for line in f:
                w, s, c = line.strip().split(',')
                item = Item(int(w), int(s), int(c))
                task.push_item(item)
    except FileNotFoundError:
        print("\n!File not found!\n")
    else:
        return task


def mutate(genes, rate):
    genes_num = len(genes)
    to_mutate_num = floor(genes_num * rate)
    to_mutate = np.random.randint(genes_num, size=to_mutate_num)
    for gene_idx in to_mutate:
        genes[gene_idx] = int(not genes[gene_idx])
    return genes


def tournament(population, tourn_size):
    indexes, fitness_arr = population.get_rand_pop_slice(tourn_size)
    best_idx = indexes[np.argmax(fitness_arr)]
    return population.get_by_idx(best_idx)


def crossover(parent1, parent2, rate):  # single point method
    if randint(0, 1) / 100 > rate:
        return parent1
    cutting_point = len(parent1) // 3
    return list(parent1[:cutting_point]) + list(parent2[cutting_point:])


# generate_task(n=1001, w=10001, s=10001, output_file='task.csv')
# task = read_task(input_file='task.csv')

def knapsack(task, POP_SIZE=100, TOURN_SIZE=10, CROSS_RATE=0.5, MUT_RATE=0.001, ITERATIONS=100):
    scores_per_gen = []

    pop = Population()
    pop.generate_rand_pop(POP_SIZE, task.n)
    i = 1
    while i < ITERATIONS:
        new_pop = []
        j = 0
        pop.calc_fitness(task)

        best, best_fit = pop.best()
        scores_per_gen.append(best_fit)
        print('Generation:', i, ', fitness:', best_fit)

        while j < POP_SIZE:
            parent1 = tournament(pop, TOURN_SIZE)
            parent2 = tournament(pop, TOURN_SIZE)
            child = crossover(parent1, parent2, CROSS_RATE)
            child = mutate(child, MUT_RATE)
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
    items = [(item.c, item.w, item.s) for item in task.items]
    values_sorted = sorted(items, key=lambda x:x[0], reverse=True)
    w_sum = 0
    c_sum = 0
    s_sum = 0
    for item in values_sorted:
        if w_sum>task.w or s_sum>task.s:
            break
        c_sum+=item[0]
        w_sum+=item[1]
        s_sum+=item[2]
    return s_sum



def crossoverTest(task, tests_num, cross_rates, iterations, **kwargs):
    print("Analysis of the crossover probability:")
    best_fit_arr = [0 for _ in range(len(cross_rates))]
    progress_fit_arr = [0 for _ in range(len(cross_rates))]
    domain = [x for x in range(1, iterations + 1)]
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1]},figsize=[10,10])
    fig.suptitle('The impact of the crossover probability on the results')
    for test_num in range(tests_num):
        i = 0
        for cross_rate in cross_rates:
            genes, fitness = knapsack(task, CROSS_RATE=cross_rate, ITERATIONS=iterations, **kwargs)
            best_fit_arr[i] += fitness[-1] // tests_num
            progress_fit_arr[i] += fitness // tests_num
            if test_num == tests_num - 1:
                axs[0].plot(domain, progress_fit_arr[i], label=str(cross_rates[i]))
            i += 1

    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Fitness')
    axs[0].legend()

    axs[1].set_xlabel('Crossover probability')
    axs[1].set_ylabel('Final fitness')
    xaxis = [str(x) for x in cross_rates]
    for i in range(tests_num):
        axs[1].bar(xaxis, best_fit_arr, width=0.1)
    #plt.show()
    plt.savefig('crossover.png')


def mutationTest(task, tests_num, mut_rates, iterations, **kwargs):
    print("Analysis of the mutation probability:")
    best_fit_arr = [0 for _ in range(len(mut_rates))]
    progress_fit_arr = [0 for _ in range(len(mut_rates))]
    domain = [x for x in range(1, iterations + 1)]
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1]},figsize=[10,10])
    fig.suptitle('The impact of the mutation probability on the results')
    for test_num in range(tests_num):
        i = 0
        for mut_rate in mut_rates:
            genes, fitness = knapsack(task, MUT_RATE=mut_rate, ITERATIONS=iterations, **kwargs)
            best_fit_arr[i] += fitness[-1] // tests_num
            progress_fit_arr[i] += fitness // tests_num
            if test_num == tests_num - 1:
                axs[0].plot(domain, progress_fit_arr[i], label=str(mut_rates[i]))
            i += 1

    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Fitness')
    axs[0].legend()

    axs[1].set_xlabel('Mutation probability')
    axs[1].set_ylabel('Final fitness')
    xaxis = [str(x) for x in mut_rates]
    for i in range(tests_num):
        axs[1].bar(xaxis, best_fit_arr, width=0.1)
    # plt.show()
    plt.savefig('mutation.png')

def tournamentTest(task, tests_num, tourn_sizes, iterations, **kwargs):
    print("Analysis of the tournament size:")
    best_fit_arr = [0 for _ in range(len(tourn_sizes))]
    progress_fit_arr = [0 for _ in range(len(tourn_sizes))]
    domain = [x for x in range(1, iterations + 1)]
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1]},figsize=[10,10])
    fig.suptitle('The impact of the tournament size on the results')
    for test_num in range(tests_num):
        i = 0
        for tourn_size in tourn_sizes:
            genes, fitness = knapsack(task, TOURN_SIZE=tourn_size, ITERATIONS=iterations, **kwargs)
            best_fit_arr[i] += fitness[-1] // tests_num
            progress_fit_arr[i] += fitness // tests_num
            if test_num == tests_num - 1:
                axs[0].plot(domain, progress_fit_arr[i], label=str(tourn_sizes[i]))
            i += 1

    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Fitness')
    axs[0].legend()

    axs[1].set_xlabel('Tournament size')
    axs[1].set_ylabel('Final fitness')
    xaxis = [str(x) for x in tourn_sizes]
    for i in range(tests_num):
        axs[1].bar(xaxis, best_fit_arr, width=0.1)
    plt.savefig('tournament.png')




def populationTest(task, tests_num, pop_sizes, iterations, **kwargs):
    print("Analysis of the population size:")
    best_fit_arr = [0 for _ in range(len(pop_sizes))]
    progress_fit_arr = [0 for _ in range(len(pop_sizes))]
    domain = [x for x in range(1, iterations + 1)]
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1]},figsize=[10,10])
    fig.suptitle('The impact of the population size on the results')
    for test_num in range(tests_num):
        i = 0
        for pop_size in pop_sizes:
            genes, fitness = knapsack(task, POP_SIZE=pop_size, ITERATIONS=iterations, **kwargs)
            best_fit_arr[i] += fitness[-1] // tests_num
            progress_fit_arr[i] += fitness // tests_num
            if test_num == tests_num - 1:
                axs[0].plot(domain, progress_fit_arr[i], label=str(pop_sizes[i]))
            i += 1

    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Fitness')
    axs[0].legend()

    axs[1].set_xlabel('Population size')
    axs[1].set_ylabel('Final fitness')
    xaxis = [str(x) for x in pop_sizes]
    for i in range(tests_num):
        axs[1].bar(xaxis, best_fit_arr, width=0.1)
    # plt.show()
    plt.savefig('population2.png')

if __name__ == '__main__':
    # generate_task(1001,10001,10001, 'task.csv')
    task = read_task('task.csv')
    # print('greedySearch:', greedySearch(task))
    # crossoverTest(task, tests_num=5, cross_rates=[0.9, 0.5, 0.1], iterations=600)
    # print('Crossover test finished')
    mutationTest(task, tests_num=1, mut_rates=[0.005, 0.002, 0.001], iterations=1000)
    print('Mutation test finished')
    # tournamentTest(task, tests_num=5, tourn_sizes=[100, 500, 900], iterations=600)
    # print('Tournament test finished')
    # populationTest(task, tests_num=1, pop_sizes=[1000,3500,6000], iterations=600)
    # print('Population test finished')
