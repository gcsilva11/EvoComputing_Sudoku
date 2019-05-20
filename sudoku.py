from random import random, sample, choice, randint
from operator import itemgetter
from collections import Counter


# Read File
def read_file(file_name):
    sudoku_list = []
    f = open(file_name, "r")
    for x in f:
        sudoku_list.append(x)
    return sudoku_list


# Split into multiple lists
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


# Map all the unchangable numbers
def get_unchangable(sudo):
    unchange_list = []
    for i in range(len(sudo)):
        for j in range(len(sudo[i])):
            if sudo[i][j] != 0:
                unchange_list.append([i, j])
    return unchange_list


# Initialize population
def init_pop(sudo, pop):
    population_list = []
    for i in range(pop):
        new_indiv = [a[:] for a in sudo]
        for j in range(len(new_indiv)):
            for k in range(len(new_indiv[j])):
                if not new_indiv[j][k]:
                    new_indiv[j][k] = randint(1, 9)

        population_list.append(new_indiv)
    return population_list


# Evaluate the sudoku - worse fitness for every number repeated on a square, on a row and on a column
def fitness_calc(sudo_to_fit):
    total_fit = 0
    # Row fitness calc
    for row in sudo_to_fit:
        c = Counter(row)
        # If there are repeated elements, increase fitness
        for value in c.values():
            total_fit += value - 1

    # Column fitness calc
    for column in zip(*sudo_to_fit):
        c = Counter(column)
        for value in c.values():
            total_fit += value - 1

    # Grid fitness calc
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            block = []
            for i in range(3):
                block.extend(sudo_to_fit[r + i][c:c + 3])

            c = Counter(block)
            for value in c.values():
                total_fit += value - 1

    return total_fit


# Parents Selection: tournament
def tour_sel(t_size):
    def tournament(pop):
        size_pop = len(pop)
        mate_pool = []
        for _ in range(size_pop):
            winner = one_tour(pop, t_size)
            mate_pool.append(winner)
        return mate_pool

    return tournament


def one_tour(population, size):
    """Maximization Problem. Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1))
    return pool[0]


def crossover(prob_cross, choice):
    # Crossover operator: Generate individuals mixing rows from parents
    def cross_rows(indi_1, indi_2):
        value = random()
        # Mutation occurs:
        if value < prob_cross:
            new_boi_1 = [indi_1[0][0], indi_2[0][1], indi_1[0][2], indi_2[0][3], indi_1[0][4], indi_2[0][5],
                         indi_1[0][6], indi_2[0][7], indi_1[0][8]]

            new_boi_2 = [indi_2[0][0], indi_1[0][1], indi_2[0][2], indi_1[0][3], indi_2[0][4], indi_1[0][5],
                         indi_2[0][6], indi_1[0][7], indi_2[0][8]]

            return (new_boi_1, 0), (new_boi_2, 0)

        else:
            return indi_1, indi_2

    # Crossover operator: Generate individuals by crossing two in a random point
    def cross_one_point(indi_1, indi_2):
        value = random()
        # Mutation occurs:
        if value < prob_cross:
            split_part = randint(0, len(indi_1[0]))

            new_boi_1 = indi_1[0][0:split_part] + indi_2[0][split_part:]
            new_boi_2 = indi_2[0][0:split_part] + indi_1[0][split_part:]

            return (new_boi_1, 0), (new_boi_2, 0)
        else:
            return indi_1, indi_2

    # Crossover operator: Generate individuals by splitting others in one point, preserving mini grids
    def cross_two_point(indi_1, indi_2):
        value = random()
        # Mutation occurs:
        if value < prob_cross:

            new_boi_1 = [indi_1[0][0], indi_1[0][1], indi_1[0][2], indi_2[0][3], indi_2[0][4], indi_2[0][5],
                         indi_1[0][6], indi_1[0][7], indi_1[0][8]]

            new_boi_2 = [indi_2[0][0], indi_2[0][1], indi_2[0][2], indi_1[0][3], indi_1[0][4], indi_1[0][5],
                         indi_2[0][6], indi_1[0][7], indi_2[0][8]]

            return (new_boi_1, 0), (new_boi_2, 0)
        else:
            return indi_1, indi_2

    if choice == 0:
        return cross_rows
    elif choice == 1:
        return cross_one_point
    else:
        return cross_two_point


def mutation(prob_muta, unchange, choice):
    # Mutation operator: Mutate individual values (if they aren't part of the initial problem)
    def mut(indiv):
        aux = [a[:] for a in indiv]
        for i in range(len(aux)):
            for j in range(len(aux[i])):
                # Mutation occurs
                if random() < prob_muta:
                    # Check if value can be changed
                    if [i, j] not in unchange:
                        aux[i][j] = randint(1, 9)

        return aux

    # Mutation operator: Swap two numbers in the same row, if they aren't part of the initial problem
    def swap(indiv):
        aux = [a[:] for a in indiv]
        # Mutation occurs
        if random() < prob_muta:
            occurred = 0
            while occurred == 0:
                # Select random row
                row = randint(0, len(aux) - 1)
                swap_1 = randint(0, 8)
                swap_2 = randint(0, 8)
                if [row, swap_1] not in unchange and [row, swap_2] not in unchange:
                    trade = aux[row][swap_1]
                    aux[row][swap_1] = aux[row][swap_2]
                    aux[row][swap_2] = trade
                    occurred = 1

        return aux

    if choice == 0:
        return mut
    else:
        return swap


# Elitism
def sel_survivors_elite(elite_ratio, choice):
    def elitism(parents, offspring):
        size = len(parents)
        comp_elite = int(size * elite_ratio)
        offspring.sort(key=itemgetter(1))
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    # applying elitism on children might make it converge too soon
    def elitism_random_children(parents, offspring):
        size = len(parents)
        comp_elite = int(size * elite_ratio)
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    if choice == 0:
        return elitism
    else:
        return elitism_random_children


# Return best population
def best_pop(populacao):
    populacao.sort(key=itemgetter(1))
    return populacao[0]


# Evolutionary Algorithm
def evolve_sudoku(choice_sudo, gens, pop_size, tournament, cross_func, muta_func, elitism):
    # Initialize Population
    pop_list = init_pop(choice_sudo, pop_size)
    # Generate new individual list with their fitness
    pop_with_fit = []
    for pop in pop_list:
        pop_with_fit.append((pop, fitness_calc(pop)))
    # Iterate
    for i in range(gens):
        # Parent Selection
        mate_pool = tournament(pop_with_fit)
        # Variation
        # Crossover
        parents = []
        for j in range(0, pop_size - 1, 2):
            indiv_1 = mate_pool[j]
            indiv_2 = mate_pool[j + 1]
            sons = cross_func(indiv_1, indiv_2)
            parents.extend(sons)
        # Mutation
        descendents = []
        for sud, fit in parents:
            new_indiv = muta_func(sud)
            descendents.append((new_indiv, fitness_calc(new_indiv)))
        # New population
        new_pop = elitism(pop_with_fit, descendents)
        # Evaluate the new population
        pop_with_fit = []
        for j in range(pop_size):
            pop_with_fit.append((new_pop[j][0], fitness_calc(new_pop[j][0])))

        print("Best Pop: " + str(best_pop(pop_with_fit)[1]) + " On Generation: " + str(i))

    return best_pop(pop_with_fit)


if __name__ == '__main__':
    SUDOKU_LEN = 81

    # Maybe run this 30 times:

    # Get all Sudokus
    filename = "sudo_Strings.txt"
    sudokus = read_file(filename)

    # Get random sudoku to solve and convert to int list
    sudoku = choice(sudokus)
    sudoku = list(map(int, sudoku[:-1]))

    # Split into rows
    real_sudoku = split_list(sudoku, 9)

    # Get unchangable positions
    unchangable_map = get_unchangable(real_sudoku)

    # print(real_sudoku)
    # [print(x[1]) for x in pop_with_fit]

    generations = 400
    pop_size = 600
    prob_muta = 0.05
    prob_cross = 0.9
    tour_size = 2
    elite_percent = 0.01
    best = evolve_sudoku(real_sudoku, generations, pop_size, tour_sel(tour_size), crossover(prob_cross, 0),
                         mutation(prob_muta, unchangable_map, 0), sel_survivors_elite(elite_percent, 1))

    print(best[1])
