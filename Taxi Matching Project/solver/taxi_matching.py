import numpy as np
import matplotlib.pyplot as plt
from random import *

def prob(p):
    return random() < p

def distance(src, dest):
    return abs(src[0] - dest[0]) + abs(src[1] - dest[1])

def crossover(parent1, parent2):
    length = len(parent1)
    
    differ = [i for i in range(length) if parent1[i] != parent2[i]]
    differ1 = [parent1[i] for i in differ]
    differ2 = [parent2[i] for i in differ]

    if(len(differ) == 0):
        return None, None
    crossbit = [randrange(len(differ))]
    
    while(differ2[crossbit[-1]] != differ1[crossbit[0]]):
        if(differ2[crossbit[-1]] == -1):
            crossbit.reverse()
            differ1, differ2 = differ2, differ1
        index = differ1.index(differ2[crossbit[-1]])
        crossbit.append(index)

    crossbit = [differ[i] for i in crossbit]

    child1 = [parent2[i] if i in crossbit else parent1[i] for i in range(length)]
    child2 = [parent1[i] if i in crossbit else parent2[i] for i in range(length)]

    return child1, child2

def mutation(individual):
    length = len(individual)
    for i in range(length):
        if(prob(1/length)):
            j = randrange(length)
            individual[i], individual[j] = individual[j], individual[i]

class Customer():
    
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest
        self.wait = 0
        self.distance = abs(src[0]-dest[0]) + abs(src[1] - dest[1])

class Taxi():

    def __init__(self, loc):
        self.loc = loc
        self.earned = 0
        self.passenger = None

    def allocate(self, customer):
        self.allocate = customer

    def drive(self):
        if self.passenger:
            direction_x = self.passenger.dest[0] - self.loc[0]
            direction_y = self.passenger.dest[1] - self.loc[1]
        
        elif self.allocate:
            direction_x = self.allocate.src[0] - self.loc[0]
            direction_y = self.allocate.src[1] - self.loc[1]
            
        else:
            return

        sign_x = 2 * (direction_x > 0) - 1
        sign_y = 2 * (direction_y > 0) - 1

        if prob(abs(direction_x)/(abs(direction_x)+abs(direction_y))):
            self.loc += sign_x
        else:
            self.loc += sign_y

        if self.passenger and self.passenger.dest == self.taxi.loc:
            self.earned += self.passenger.distance
            self.passenger = None
        elif self.allocate and self.allocate.src == self.taxi.loc:
            self.passenger = self.allocate
            self.allocate = None

class World():

    def __init__(self, world_size, random=True, **kwargs):
        def rand_axis():
            return (randrange(world_size[0]), randrange(world_size[1]))
            
        self.world_size = world_size
        self.num_customers = 0
        self.customers = []

        if random:
            self.customers = [Customer(rand_axis(), rand_axis()) for i in range(kwargs['num_customers'])]
            self.taxis = [Taxi(rand_axis()) for i in range(kwargs['num_taxis'])]
            self.num_customers = kwargs['num_customers']
            self.num_taxis = kwargs['num_taxis']
        
        self.solutions = []
        self.complete_sol = []
        self.greedy_sol = []
        self.genetic_sol = []

    def add(self, src, dest):
        self.customers.append(Customer(src, dest))
        self.num_customers += 1

    def taxis(self, taxis):
        self.taxis = [Taxi(loc) for loc in taxis]
        self.num_taxis = len(taxis)

    def fitness(self, solution_type="", print_option=False):
        waiting_time = []
        earn = []

        if solution_type == "complete":
            solution = self.complete_sol
        elif solution_type == "greedy":
            solution = self.greedy_sol
        elif solution_type == "genetic":
            solution = self.genetic_sol
        else:
            solution = self.solutions

        for tid, cid in enumerate(solution):
            taxi = self.taxis[tid]

            if cid < 0:
                earn.append(taxi.earned)
                continue

            customer = self.customers[cid]
            passenger = taxi.passenger
            
            if passenger == None:
                waiting_time.append(customer.wait + distance(customer.src, taxi.loc))
                earn.append(taxi.earned + customer.distance)
            else:
                waiting_time.append(customer.wait + distance(taxi.loc, passenger.dest) + distance(passenger.dest, customer.src))
                earn.append(taxi.earned + passenger.distance + customer.distance)
        
        if print_option:
            print(f"waiting time : {waiting_time}")
            print(f"total waiting time : {sum(waiting_time)}")
            print(f"variance of waiting time : {np.var(waiting_time)}")
            print(f"variance of earn : {np.var(earn)}")
        return (sum(waiting_time) ** 2) + np.var(waiting_time) + np.var(earn)

    def complete(self):
        self.complete_sol = [min(i, len(self.customers)-1) for i in range(len(self.taxis))]
        min_fit = self.fitness("complete")

        def complete_r(sol):
            if(len(sol) == len(self.taxis)):
                nonlocal min_fit

                self.solutions = sol
                fit = self.fitness("")
                if(fit < min_fit):
                    min_fit = fit
                    self.complete_sol = sol[:]

                sol.pop()
                return

            remain = set(range(len(self.customers))) - set(sol)
            if(len(sol) + len(remain) < len(self.taxis)):
                remain.add(-1)
            
            for c in remain:
                sol.append(c)
                complete_r(sol)

            if(len(sol) > 0):
                sol.pop()
            return

        complete_r([])
        self.complete_fit = min_fit
        return self.complete_sol

    def greedy(self):
        taxi_remain = list(range(len(self.taxis)))
        sol = []

        def waitTime(tid, cid):
            taxi = self.taxis[tid]
            customer = self.customers[cid]
            if taxi.passenger == None:
                return distance(taxi.loc, customer.src)
            else:
                return distance(taxi.loc, taxi.passenger.dest) + (taxi.passenger.dest, customer.src)

        for cid in range(len(self.customers)):
            min_tid = min([(tid, waitTime(tid, cid)) for tid in taxi_remain], key=lambda x:x[1])[0]
            sol.append((min_tid, cid))
            taxi_remain.remove(min_tid)

        for tid in taxi_remain:
            sol.append((tid, -1))

        self.greedy_sol = [cid for _, cid in sorted(sol)]
        self.greedy_fit = self.fitness("greedy")
        return self.greedy_sol

    def genetic(self, monitor=False, graph=False, **kwargs):
        max_generation = kwargs['max_generation']
        selected_num = kwargs['selected_num']
        max_population = selected_num * (selected_num + 1)
        
        #customer_set = list(range(len(self.customers))) + [-1] * (len(self.taxis) - len(self.customers))
        solutions = []
        fit_generation = []

        def select():
            return sorted(solutions, key=lambda x:x[1])[:selected_num]

        def remove_duplicates():
            duplicates = []
            for i in range(len(solutions)):
                for j in range(i+1, len(solutions)):
                    if solutions[i][0] == solutions[j][0]:
                        duplicates.append(j)
            
            duplicates = list(set(duplicates))
            for j in sorted(duplicates, reverse=True):
                solutions.pop(j)

        self.greedy()
        for i in range(max_population):
            sol = self.greedy_sol[:]
            mutation(sol)
            self.solutions = sol
            solutions.append((sol, self.fitness()))

        remove_duplicates()

        for i in range(max_generation):
            selected = select()
            solutions = selected[:]
            if monitor and i%(max_generation // 10) == 0:
                print(f"<< GENERATION {i} >>", end='\n\n')
                
            for j in range(selected_num):
                self.solutions = selected[j][0]
                if monitor and i%(max_generation // 10) == 0:
                    print(f"** individual {j} **")
                    print(f"solution : {selected[j][0]}")
                    print(f"fitness : {self.fitness(print_option=False)}", end='\n\n')
            self.genetic_sol = selected[0][0]
            fit_generation.append(self.fitness("genetic"))
            if monitor:
                self.show("genetic")
            
            for pid1, (parent1, _) in enumerate(selected):
                for parent2, _ in selected[pid1+1:]:
                    #print("crossover...")
                    child1, child2 = crossover(parent1, parent2)
                    if(child1 == None or child2 == None):
                        continue

                    #print("mutations...")
                    mutation(child1)
                    mutation(child2)

                    self.solutions = child1
                    solutions.append((child1, self.fitness()))
                    self.solutions = child2
                    solutions.append((child2, self.fitness()))

                    remove_duplicates()

        if graph:
            plt.plot(fit_generation, label='genetic', linewidth=3)
            plt.plot([self.greedy_fit]*max_generation, '--', label='greedy')
            if not self.complete_sol == []:
                plt.plot([self.complete_fit]*max_generation, '--', label='complete')

            #plt.xticks(range(max_generation))
            plt.xlabel('Generation')
            plt.ylabel('fitness')

            plt.legend()
            plt.show()

        return self.genetic_sol
        

    def show(self, solution_type):
        if solution_type == "complete":
            solution = self.complete_sol
        elif solution_type == "greedy":
            solution = self.greedy_sol
        elif solution_type == "genetic":
            solution = self.genetic_sol
        else:
            solution = self.solutions

        if not solution_type == "plain":
            print("fitness : ", self.fitness(solution_type, print_option=True))

        plt.figure(figsize = (20,20))
        plt.scatter([c.src[0] for c in self.customers], [c.src[1] for c in self.customers], c='b')
        plt.scatter([t.loc[0] for t in self.taxis], [t.loc[1] for t in self.taxis], c='r')

        for i, c in enumerate(self.customers):
            plt.text(c.src[0]+0.2, c.src[1]+0.2, 'C'+str(i))
        for i, t in enumerate(self.taxis):
            plt.text(t.loc[0]+0.2, t.loc[1]+0.2, 'T'+str(i))
        
        if not solution_type == "plain":
            for tid, cid in enumerate(solution):
                if cid == -1:
                    continue
                if self.taxis[tid].passenger == None:
                    plt.annotate('', xy=self.customers[cid].src, xytext=self.taxis[tid].loc, arrowprops={'color':'green', 'width':0.5})
                else:
                    plt.annotate('', xy=self.taxis[tid].passenger.dest, xytext=self.taxis[tid].loc, arrowprops={'color':'green', 'width':0.5})
                    plt.annotate('', xy=self.customers[cid].src, xytext=self.taxis[tid].passenger.dest, arrowprops={'color':'green', 'width':0.5})

        plt.xticks(range(self.world_size[0]))
        plt.yticks(range(self.world_size[1]))
        plt.grid()
        plt.show()
