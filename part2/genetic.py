from deap import base, creator, tools, algorithms, gp

import sys
import math
import random
import operator
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv


warnings.filterwarnings("ignore")


def read_data(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        next(f)
        next(f)
        for line in f:
            line_split = line.strip().split(' ')
            x.append(float(line_split[0]))
            y.append(float(line_split[-1]))
    return x, y


data_x, data_y = read_data('regression.txt')
data_x = np.array(data_x)
data_y = np.array(data_y)


# Protected logarithm function, returns 0 if argument is very small less than 0.001, else returns the logarithm of the
# absolute value of the argument.
def protected_log(left):
    if left < 0.001:
        return 0
    else:
        return math.log(abs(left))


# Protected square root function, returns 0 if argument is very small less than 0.001, else returns the square root of
# the absolute value of the argument.
def protected_sqrt(left):
    return math.sqrt(abs(left))


# Protected inverse function, returns 0 if argument is between -0.001 and 0.001, else returns the inverse of the
# argument.
def protected_inv(left):
    if -0.001 <= left <= 0.001:
        return 0
    return 1 / left


# Protected Division function, returns 0 if any of the arguments is very small between -0.001 and 0.001, else returns
# the division of the numerator and the denominator.
def protected_div(left, right):
    if (-0.001 <= left <= 0.001) or (-0.001 <= right <= 0.001):
        return 0
    return left / right


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(operator.neg, 1)
#pset.addPrimitive(math.cos, 1)
#pset.addPrimitive(math.sin, 1)
#pset.addPrimitive(math.tan, 1)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(abs, 1)
# pset.addPrimitive(min, 2)
# pset.addPrimitive(max, 2)
pset.addPrimitive(protected_sqrt, 1)
pset.addPrimitive(protected_log, 1)
pset.addPrimitive(protected_inv, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0='x')


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    # sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    sqerrors = 0
    for i, x in enumerate(points):
        try:
            sqerrors += (func(x) - data_y[i]) ** 2
        except OverflowError:
            sqerrors += sys.float_info.max

    return sqerrors / len(points),


toolbox.register("evaluate", evalSymbReg, points=data_x)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(318)

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    crossover_rate = 0.9
    mutation_rate = 0.1
    n_gen = 300

    algorithms.eaSimple(pop, toolbox, crossover_rate, mutation_rate, n_gen, stats, halloffame=hof, verbose=True)
    print("\nBest individual is ", hof[0])
    print("\nBest fitness is ", hof[0].fitness)

    expr = hof[0]
    nodes, edges, labels = gp.graph(expr)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.png")

    return pop, stats, hof


if __name__ == "__main__":
    main()
