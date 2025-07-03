"""Evolutionary Algorihm Based Optimizations."""

import random
from functools import partial

import numpy as np
from deap import algorithms, base, creator, tools
from numpy.typing import NDArray


def lambda_plus_mu(
    n_modes: int,
    evaluate: partial[list[float]],
    pop_size: int,
    ngen: int,
) -> NDArray:
    """Reorder modes using lambda+mu evolutionary algorithm.

    The values of lambda and mu are both set to half of the population size.
    With some tinkering this seems to work well.

    Args:
        n_modes (int): The number of modes in the system.
        evaluate (partial[list[float]]): A partial function which acts as the cost function.
            This should take a list of integers (mode labels) as input
            and output a list of floats.
        pop_size (int): The size of the initial population.
        ngen (int): The number of generations to evolve.

    Returns:
        NDArray: The best mode ordering found.

    Example:
        >>> import numpy as np
        >>> from functools import partial
        >>> from ferrmion.optimize.enumeration.evolutionary import lambda_plus_mu
        >>> def dummy_eval(x): return [sum(x)]
        >>> evaluate = partial(dummy_eval)
        >>> lambda_plus_mu(3, evaluate, pop_size=10, ngen=2)
    """
    size = n_modes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(size), size)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("evaluate", evaluate)

    def initIndividual(icls, content):
        return icls(content)

    def initPopulation(pcls, ind_init, first_gen):
        return pcls(ind_init(c) for c in first_gen)

    toolbox.register("individual_guess", initIndividual, creator.Individual)

    first_gen = toolbox.population(n=pop_size)[1:]
    first_gen.append([*range(size)])
    toolbox.register(
        "population_guess", initPopulation, list, toolbox.individual_guess, first_gen
    )
    population = toolbox.population_guess()

    toolbox.register("evaluate", evaluate)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)
    pop, logbook = algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=pop_size // 2,
        lambda_=pop_size // 2,
        cxpb=0.02,
        mutpb=0.6,
        ngen=ngen,
        halloffame=hof,
        stats=stats,
        verbose=False,
    )
    return np.array(hof[0]), logbook
