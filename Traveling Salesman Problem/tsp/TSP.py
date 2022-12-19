import sys
import getopt
from random import *
from math import sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt
        
coords = []

def distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def prob(p):
    r = random()
    return r < p

def remove_duplicate(solutions):
    unique = []
    duplicate = 0
    for solution in solutions:
        for u in unique:
            if u.sol == solution.sol:
                duplicate = 1
                break
        
        if not duplicate:
            unique.append(solution)
    
    return unique

class Solution():
    def __init__(self, sol):
        self.sol = sol
        self.length = len(self.sol)
        self.standardize()
        self.fitness()

    def standardize(self):
        index = self.sol.index(0)
        self.sol = self.sol[index:] + self.sol[:index]

    def fitness(self):
        self.fit_val = 0
        self.distances = []
        for i, (a, b) in enumerate(zip(self.sol, self.sol[1:] + [self.sol[0]])):
            d = distance(coords[a], coords[b])
            self.fit_val += d
            self.distances.append(d ** 2)
        
        self.sum_distance = sum(self.distances)

    def choose_mutate(self, i):
      probability = []
      for j in range(self.length):
        close = distance(coords[self.sol[i]], coords[self.sol[j]])
        if close == 0:
          probability.append(0)
        else:
          probability.append(self.distances[j] / close ** 2)

      prob_sum = sum(probability)
      probability = [p / prob_sum for p in probability]

      p = random()

      acc = 0
      for j in range(self.length):
        acc += probability[j]
        if acc > p:
          # print(f"{self.sol[j]} selected with probability {probability[j]} from {self.sol[i]}")
          return j

    def mutate(self):
        for i in range(self.length):
          if prob(self.distances[i] / self.sum_distance):
              # print(f"{self.sol[i]} selected with probability {self.distances[i] / self.sum_distance}")

              j = self.choose_mutate(i)
              # print(f"mutate {self.sol[i]} and {self.sol[j]}")

              rev = self.sol[min(i, j)+1 : max(i, j)+1]
              rev.reverse()
              self.sol[min(i, j)+1 : max(i, j)+1] = rev


        self.fitness()
        self.standardize()

    def crossover(self, Sol2):
        crosspoint1 = randrange(0, self.length)
        crosspoint2 = randrange(0, self.length)

        i = min(crosspoint1, crosspoint2)
        j = max(crosspoint1, crosspoint2)

        check = [0] * self.length  
        child1 = self.sol[i:j]
        for c in child1:
          check[c] = 1
        index = Sol2.sol.index(self.sol[j])
        for s in Sol2.sol[index:] + Sol2.sol[:index]:
            if check[s] == 0:
                child1.append(s)

        check = [0] * self.length
        child2 = Sol2.sol[i:j]
        for c in child2:
          check[c] = 1
        index = self.sol.index(Sol2.sol[j])
        for s in self.sol[index:] + self.sol[:index]:
            if check[s] == 0:
                child2.append(s)

        Child1 = Solution(child1)
        Child2 = Solution(child2)

        Child1.mutate()
        Child2.mutate()

        return Child1, Child2

def greedy(seed, dimension):
    greedy = [seed]
    remain = list(range(dimension))
    remain.remove(seed)

    while len(remain) > 0:
        next = min([(r, distance(coords[r], coords[greedy[-1]])) for r in remain],
                key= lambda x:x[1])[0]
        greedy.append(next)
        remain.remove(next)

    return greedy

def upload_solution(log_file):
    population = []
    upload = open(log_file, "r")

    line = upload.readline()
    while line != "":
        prev_sol = line.split(",")
        prev_sol = [int(s) for s in prev_sol]
        population.append(Solution(prev_sol))

        line = upload.readline()

    upload.close()

    return population

def read_problem(file_name):
    global coords

    tsp = open(file_name, "r")

    line = tsp.readline()
    while not "NODE_COORD_SECTION" in line:
        feature = line.split(":")[0].strip()
        if feature in ["NAME", "COMMENT", "EDGE_WEIGHT_TYPE", "TYPE"]:
            pass
        elif feature == "DIMENSION":
            dimension = int(line.split(":")[1].strip())
        else:
            print("unknown feature: ", line)

        line = tsp.readline()

    for i in range(dimension):
        _, x, y = tsp.readline().split()
        coords.append((float(x),float(y)))

    line = tsp.readline()
    if(line.strip() == "EOF"): pass
    else: print("still remain: ", line)

    tsp.close()
    return dimension

def Solve(population = None, fit_generation = [], **kwargs):

    dimension = kwargs["dimension"]
    max_population = kwargs["max_population"]
    max_generation = kwargs["max_generation"]
    show_fitness = kwargs["show_fitness"]
    selected = max_population // 2

    children = []

    if population == None:
        population = []
        seed = greedy(randrange(dimension), dimension)
        Seed = Solution(seed)
        population.append(Seed)

    progress = tqdm(range(max_generation))

    for i in progress:
        population = remove_duplicate(population)
        children = remove_duplicate(children)
        population.sort(key=lambda sol: sol.fit_val)
        children.sort(key=lambda sol: sol.fit_val)

        progress.set_description(f"fitness value {population[0].fit_val}")

        population = population[:selected // 2] + children[:selected // 2]
        children = []

        parents1 = population[:]
        parents2 = population[:]
        shuffle(parents1)
        shuffle(parents2)

        for p1, p2 in zip(parents1, parents2):
            child1, child2 = p1.crossover(p2)
            children.append(child1)
            children.append(child2)

        fit_generation.append(min(population, key= lambda sol: sol.fit_val).fit_val)

    if show_fitness:
      plt.plot(fit_generation, label='fitness value', linewidth=3)

      plt.xlabel('Generation')
      plt.ylabel('fitness')

      plt.legend()
      plt.show()

    return population, fit_generation

def main():

    max_population = 100
    max_generation = 1000
    solution_log = None
    population = None
    fit_generation = []
    repeat = 1
    show_fitness = False

    oplist, args = getopt.getopt(sys.argv[1:], "p:g:sur:", ["save=", "upload=", "repeat=", "show-fitness"])

    dimension = read_problem(args[0])

    for op, arg in oplist:
        if op == "-p":
            max_population = int(arg)
        elif op == "-g":
            max_generation = int(arg)
        elif op in ['-s', '--save']:
            if arg == '':
                solution_log = 'solution_log.csv'
            else:
                solution_log = arg
        elif op in ['-u', '--upload']:
            if arg == '':
                population = upload_solution("solution_log.csv")
            else:
                population = upload_solution(arg)
        elif op in ["-r", "--repeat"]:
            repeat = arg
        elif op == "--show-fitness":
            show_fitness = True

    population, fit_generation = Solve(population=population, fit_generation= fit_generation,
                                       max_population = max_population, 
                                       max_generation = max_generation, 
                                       dimension = dimension,
                                       show_fitness = show_fitness)
    for i in range(repeat - 1):
        population,fit_generation = Solve(population= population, fit_generation= fit_generation,
                                          max_population = max_population,
                                          max_generation = max_generation,
                                          dimension = dimension,
                                          show_fitness = False)

    if not solution_log == None:
        log = open(solution_log, "w")
        for S in population:
            sol = [str(s) for s in S.sol]
            log.write(",".join(sol))
            log.write("\n")
        log.close()

    solution = min(population, key= lambda sol: sol.fit_val).sol
    for s in solution:
        print(s+1)

main()