from random import random, sample, choice, randint, shuffle, seed
from operator import itemgetter
from collections import Counter


# Split into multiple lists
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


# Map all the fixed numbers
def get_fixed(sudo):
    fixed_list = []
    for i in range(len(sudo)):
        for j in range(len(sudo[i])):
            if sudo[i][j] != 0:
                fixed_list.append([i, j])
    return fixed_list


# Initialize population
def init_pop(sudo, perm, disallowed):
    def pop_perm(pop):
        population_list = []
        for i in range(pop):
            new_indiv = []
            for line in sudo:
                available = list(range(1, 10))
                for square in line:
                    if square != 0:
                        available.remove(square)
                shuffle(available)
                new_indiv.append(available)

            population_list.append(new_indiv)
        return population_list

    def pop_int(pop):
        population_list = []
        for i in range(pop):
            new_indiv = [a[:] for a in sudo]
            for j in range(len(new_indiv)):
                for k in range(len(new_indiv[j])):
                    if new_indiv[j][k] == 0:
                        allowed = [l for l in range(1, 10) if l not in disallowed[j][k]]
                        new_indiv[j][k] = choice(allowed)

            population_list.append(new_indiv)
        return population_list

    if perm:
        return pop_perm
    else:
        return pop_int


# Evaluate the sudoku - worse fitness for every number repeated on a square, on a row and on a column
def fitness_func(fixed_map, perm, real_sudoku):
    def fitness_int(sudo):
        total_fit = 0
        # Row fitness calc
        for i in range(len(sudo)):
            c = Counter(sudo[i])
            # If there are repeated elements, increase fitness
            for value in c.values():
                total_fit += value - 1

        # Column fitness calc
        for column in zip(*sudo):
            c = Counter(column)
            for value in c.values():
                total_fit += value - 1

        # Grid fitness calc
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                block = []
                for i in range(3):
                    block.extend(sudo[r + i][c:c + 3])

                c = Counter(block)
                for value in c.values():
                    total_fit += value - 1

        return total_fit

    def fitness_perm(sudo):
        sudo = [line[:] for line in sudo]
        for line, col in fixed_map:
            sudo[line].insert(col, real_sudoku[line][col])

        total_fit = 0
        for column in zip(*sudo):
            c = Counter(column)
            for value in c.values():
                total_fit += value - 1
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                block = []
                for i in range(3):
                    block.extend(sudo[r + i][c:c + 3])

                c = Counter(block)
                for value in c.values():
                    total_fit += value - 1

        return total_fit

    if perm:
        return fitness_perm
    else:
        return fitness_int


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


def crossover(prob_cross, pick):
    """CROSSOVERS FOR INTEGERS"""

    # Crossover operator: Generate individuals mixing rows from parents
    def cross_rows(indi_1, indi_2):
        value = random()
        # Crossover occurs:
        if value < prob_cross:
            new_indi_1 = [indi_1[0][0], indi_2[0][1], indi_1[0][2], indi_2[0][3], indi_1[0][4], indi_2[0][5],
                          indi_1[0][6], indi_2[0][7], indi_1[0][8]]

            new_indi_2 = [indi_2[0][0], indi_1[0][1], indi_2[0][2], indi_1[0][3], indi_2[0][4], indi_1[0][5],
                          indi_2[0][6], indi_1[0][7], indi_2[0][8]]

            return (new_indi_1, 0), (new_indi_2, 0)

        else:
            return indi_1, indi_2

    # Crossover operator: Generate individuals by crossing two in a random point
    def cross_one_point(indi_1, indi_2):
        value = random()
        # Crossover occurs:
        if value < prob_cross:
            split_part = randint(0, len(indi_1[0]))

            new_indi_1 = indi_1[0][0:split_part] + indi_2[0][split_part:]
            new_indi_2 = indi_2[0][0:split_part] + indi_1[0][split_part:]

            return (new_indi_1, 0), (new_indi_2, 0)
        else:
            return indi_1, indi_2

    # Crossover operator: Generate individuals by splitting others in one point, preserving mini grids
    def cross_two_point(indi_1, indi_2):
        value = random()
        # Crossover occurs:
        if value < prob_cross:

            new_indi_1 = [indi_1[0][0], indi_1[0][1], indi_1[0][2], indi_2[0][3], indi_2[0][4], indi_2[0][5],
                          indi_1[0][6], indi_1[0][7], indi_1[0][8]]

            new_indi_2 = [indi_2[0][0], indi_2[0][1], indi_2[0][2], indi_1[0][3], indi_1[0][4], indi_1[0][5],
                          indi_2[0][6], indi_1[0][7], indi_2[0][8]]

            return (new_indi_1, 0), (new_indi_2, 0)
        else:
            return indi_1, indi_2

    """CROSSOVERS FOR PERMUTATIONS"""

    def order_cross(cromo_1, cromo_2):
        if random() < prob_cross:
            cromo_1 = cromo_1[0]
            cromo_2 = cromo_2[0]
            new_indi_1 = []
            new_indi_2 = []
            for a in range(len(cromo_1)):
                aux1 = cromo_1[a]
                aux2 = cromo_2[a]
                size = len(aux1)
                if size < 2:
                    new_indi_1.append(aux1)
                    new_indi_2.append(aux2)
                    continue
                pc = sample(range(size), 2)
                pc.sort()
                pc1, pc2 = pc
                f1 = [None] * size
                f2 = [None] * size
                f1[pc1:pc2 + 1] = aux1[pc1:pc2 + 1]
                f2[pc1:pc2 + 1] = aux2[pc1:pc2 + 1]
                for j in range(size):
                    for i in range(size):
                        if (aux2[j] not in f1) and (f1[i] is None):
                            f1[i] = aux2[j]
                            break
                    for k in range(size):
                        if (aux1[j] not in f2) and (f2[k] is None):
                            f2[k] = aux1[j]
                            break
                new_indi_1.append(f1)
                new_indi_2.append(f2)
            return (new_indi_1, 0), (new_indi_2, 0)
        else:
            return cromo_1, cromo_2

    def pmx_cross(cromo_1, cromo_2):
        if random() < prob_cross:
            cromo_1 = cromo_1[0]
            cromo_2 = cromo_2[0]
            new_indi_1 = []
            new_indi_2 = []
            for a in range(len(cromo_1)):
                aux1 = cromo_1[a]
                aux2 = cromo_2[a]
                size = len(aux1)
                if size < 2:
                    new_indi_1.append(aux1)
                    new_indi_2.append(aux2)
                    continue
                pc = sample(range(size), 2)
                pc.sort()
                pc1, pc2 = pc
                f1 = [None] * size
                f2 = [None] * size
                f1[pc1:pc2 + 1] = aux1[pc1:pc2 + 1]
                f2[pc1:pc2 + 1] = aux2[pc1:pc2 + 1]
                # primeiro filho
                # parte do meio
                for j in range(pc1, pc2 + 1):
                    if aux2[j] not in f1:
                        pos_2 = j
                        g_j_2 = aux2[pos_2]
                        g_f1 = f1[pos_2]
                        index_2 = aux2.index(g_f1)
                        while f1[index_2] is not None:
                            index_2 = aux2.index(f1[index_2])
                        f1[index_2] = g_j_2
                # restantes
                for k in range(size):
                    if f1[k] is None:
                        f1[k] = aux2[k]
                # segundo filho
                # parte do meio
                for j in range(pc1, pc2 + 1):
                    if aux1[j] not in f2:
                        pos_1 = j
                        g_j_1 = aux1[pos_1]
                        g_f2 = f2[pos_1]
                        index_1 = aux1.index(g_f2)
                        while f2[index_1] is not None:
                            index_1 = aux1.index(f2[index_1])
                        f2[index_1] = g_j_1
                # parte restante
                for k in range(size):
                    if f2[k] is None:
                        f2[k] = aux1[k]
                new_indi_1.append(f1)
                new_indi_2.append(f2)
            return (new_indi_1, 0), (new_indi_2, 0)
        else:
            return cromo_1, cromo_2

    if pick == 0:
        return cross_rows
    elif pick == 1:
        return cross_one_point
    elif pick == 2:
        return cross_two_point
    elif pick == 3:
        return order_cross
    else:
        return pmx_cross


def mutation(prob_muta, fixed, disallowed, pick):
    """MUTATIONS FOR INTEGERS"""

    # Mutation operator: Mutate individual values (if they aren't part of the initial problem)
    def mut(indiv):
        indiv = [a[:] for a in indiv]
        for i in range(len(indiv)):
            for j in range(len(indiv[i])):
                # Check if value can be changed
                if [i, j] in fixed:
                    continue
                # Mutation occurs
                if random() < prob_muta:
                    allowed = [l for l in range(1, 10) if l not in disallowed[i][j]]
                    if len(allowed) > 1:
                        aux = indiv[i][j]
                        while indiv[i][j] == aux:
                            indiv[i][j] = choice(allowed)
        return indiv

    """MUTATIONS FOR PERMUTATIONS"""

    def muta_perm_swap(indiv):
        indiv = [a[:] for a in indiv]
        if random() < prob_muta:
            for line in indiv:
                size = len(line)
                if size < 2:
                    continue
                index = sample(range(size), 2)
                index.sort()
                i1, i2 = index
                line[i1], line[i2] = line[i2], line[i1]
        return indiv

    def muta_perm_scramble(indiv):
        indiv = [a[:] for a in indiv]
        if random() < prob_muta:
            for line in indiv:
                size = len(line)
                if size < 2:
                    continue
                index = sample(range(size), 2)
                index.sort()
                i1, i2 = index
                scramble = line[i1:i2 + 1]
                shuffle(scramble)
                line[i1:i2 + 1] = scramble
        return indiv

    def muta_perm_insertion(indiv):
        indiv = [a[:] for a in indiv]
        if random() < prob_muta:
            for line in indiv:
                size = len(line)
                if size < 2:
                    continue
                index = sample(range(size), 2)
                index.sort()
                i1, i2 = index
                gene = line[i2]
                for i in range(i2, i1, -1):
                    line[i] = line[i - 1]
                line[i1 + 1] = gene
        return indiv

    if pick == 0:
        return mut
    elif pick == 1:
        return muta_perm_swap
    elif pick == 2:
        return muta_perm_scramble
    else:
        return muta_perm_insertion


# Elitism
def sel_survivors_elite(elite_ratio):
    # sorting children might make it converge too soon
    def elitism_random_children(parents, offspring):
        size = len(parents)
        comp_elite = int(size * elite_ratio)
        shuffle(offspring)
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    return elitism_random_children


# Return best population
def best_pop(populacao):
    populacao.sort(key=itemgetter(1))
    return populacao[0]


# Evolutionary Algorithm
def evolve_sudoku(init_func, gens, pop_size, fitness, tournament, cross_func, muta_func, elitism):
    # Initialize Population
    pop_list = init_func(pop_size)
    # Generate new individual list with their fitness
    pop_with_fit = []
    for pop in pop_list:
        pop_with_fit.append((pop, fitness(pop)))
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
            descendents.append((new_indiv, fitness(new_indiv)))
        # New population
        pop_with_fit = elitism(pop_with_fit, descendents)
        current_best = best_pop(pop_with_fit)
        print("Best Pop: " + str(current_best[1]) + " On Generation: " + str(i))
        if current_best[1] == 0:
            break
    count = 0
    for indiv in pop_with_fit:
        count += indiv[1]
    return best_pop(pop_with_fit), i, count / pop_size


if __name__ == '__main__':
    # Fixed variables
    generations = 10
    population_size = 10
    prob_mutation = 0.05
    prob_crossover = 0.7
    tournament_size = 2
    elite_percent = 0.01
    combs = [
        (False, 0, 0),
        (False, 1, 0),
        (False, 2, 0),
        (True, 3, 1),
        (True, 3, 2),
        (True, 3, 3),
        (True, 4, 1),
        (True, 4, 2),
        (True, 4, 3),
    ]
    # Get all Sudokus
    filename = "sudoku.txt"
    with open(filename, "r") as f:
        lines = f.readlines()

    # Get random sudoku to solve and convert to int list
    for n in range(len(lines)):
        sudoku = lines[n]
        if sudoku[-1] == '\n':
            sudoku = sudoku[:-1]
        sudoku_l = list(map(int, sudoku))

        real_sudoku = split_list(sudoku_l, 9)  # Split into rows
        fixed_map = get_fixed(real_sudoku)  # Get fixed positions

        zipped_og = list(zip(*real_sudoku))
        block_og = []
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                block = []
                for i in range(3):
                    block.extend(real_sudoku[r + i][c:c + 3])
                block_og.append(block)
        disallowed_set = [[None for _ in range(9)] for _ in range(9)]
        for i in range(len(real_sudoku)):
            for j in range(len(real_sudoku[i])):
                disallowed_set[i][j] = set(real_sudoku[i]).union(set(zipped_og[j])).union(set(block_og[(i // 3) * 3 + j // 3]))

        for comb in combs:
            permutation = comb[0]
            crossover_pick = comb[1]
            mutation_pick = comb[2]

            best_overall = [0, 1000]
            avg_best = 0
            best_gens = 1000
            avg_gens = 0
            avg_avg = 0
            for i in range(30):
                seed(i * 100)
                best_indiv, gens, avg = \
                    evolve_sudoku(init_pop(real_sudoku, permutation, disallowed_set),generations,
                                  population_size, fitness_func(fixed_map, permutation, real_sudoku),
                                  tour_sel(tournament_size), crossover(prob_crossover, crossover_pick),
                                  mutation(prob_mutation, fixed_map, disallowed_set, mutation_pick),
                                  sel_survivors_elite(elite_percent))

                if best_indiv[1] < best_overall[1]:
                    best_overall = best_indiv
                if gens < best_gens:
                    best_gens = gens
                avg_best += best_indiv[1]
                avg_gens += gens
                avg_avg += avg

            with open(str(n)+".txt", "a") as f:
                print("{},{},{},{},{},{},{},{}".format(permutation, crossover_pick, mutation_pick, best_overall[1],
                                                       avg_best/30, best_gens, avg_gens/30, avg_avg/30), file=f)
