from random import randint
from math import floor


class Individual:
    def __init__(self, genes_num=None, genes=None):
        if genes is None:
            self.genes = [randint(0,1) for _ in range(genes_num)]
        else:
            self.genes = genes

    def mutate(self, rate):
        genes_num = len(self.genes)
        to_mutate_num = floor(genes_num * rate)
        to_mutate = []
        for i in range(to_mutate_num):
            gene_idx = randint(0, genes_num - 1)
            while gene_idx in to_mutate:
                gene_idx = gene_idx + 1 if gene_idx < genes_num-1 else 0
            to_mutate.append(gene_idx)

        for gene_idx in to_mutate:
            self.genes[gene_idx] = int(not self.genes[gene_idx])


class Population:
    def __init__(self, pop_size=0, genes_num=None):
        self.population = [Individual(genes_num) for _ in range(pop_size)]

    def push_ind(self, ind):
        self.population.append(ind)

    def best(self, task):
        fitnesses = [fitness(indiv, task) for indiv in self.population]
        index = fitnesses.index(max(fitnesses))
        return self.population[index]

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


def fitness(individual, task):
    max_w = task.w
    max_s = task.s
    score = 0
    w_sum = 0
    s_sum = 0
    for is_present, item in zip(individual.genes, task.items):
        if is_present:
            score += item.c
            w_sum += item.w
            s_sum += item.s
            if w_sum > max_w or s_sum > max_s:
                score = -1000
                break
    return score


def tournament(population, tourn_size, task):
    pop_size = len(population)
    participants = []
    for i in range(tourn_size):
        candidate = randint(0, pop_size - 1)
        while candidate in participants:
            candidate = candidate + 1 if candidate < pop_size-1 else 0
        participants.append(candidate)

    pop_slice = [population[participant] for participant in participants]
    fitnesses = [fitness(indiv, task) for indiv in pop_slice]
    max_fitness = max(fitnesses)
    index = fitnesses.index(max_fitness)
    return pop_slice[index]


def crossover(parent1, parent2, rate):
    if randint(0, 1) / 100 > rate:
        return parent1
    cutting_point = len(parent1.genes) // 3
    child_genes = parent1.genes[:cutting_point] + parent2.genes[cutting_point:]
    return Individual(genes=child_genes)



if __name__ == '__main__':
    POP_SIZE = 1000
    TOURN_SIZE = 400
    CROSS_RATE = 0.4
    MUT_RATE = 0.1
    generate_task(n=1001, w=10001, s=10001, output_file='task.csv')
    task = read_task(input_file='task.csv')
    pop = Population(pop_size=POP_SIZE, genes_num=task.n)
    i = 0
    while i<100:
        print('Generation:', i+1)
        new_pop = Population()
        j = 0
        while j<POP_SIZE:
            parent1 = tournament(pop.population,TOURN_SIZE, task)
            parent2 = tournament(pop.population,TOURN_SIZE, task)
            child = crossover(parent1,parent2,CROSS_RATE)
            child.mutate(MUT_RATE)
            new_pop.push_ind(child)
            j+=1
        pop = new_pop
        thebest = pop.best(task)
        print(fitness(thebest, task), sum(thebest.genes),thebest.genes)
        i+=1


