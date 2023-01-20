from game import population
from utils_costs import *

def main(
        n_strategies=3, utils_costs='const',
        k1=20, k2=5,
        prob_by_strategy_mod=prob_by_strategy_mod_null,
        population_size=2000, n_iter=100,
        learning_rate=.05, debug=False
    ):

    utils, costs = utils_costs_const(n_strategies, k1, k2) if utils_costs == 'const' else utils_costs_exp(n_strategies, k1, k2)

    game = population(costs, prob_by_strategy_mod, utils, population_size, n_iter, learning_rate, debug)

    game.compute_payoffs()

    game.print_stats()
    game.print_payoffs()

    game.plot_strategies(True)
    #game.plot_standard_deviation_strategies()
    game.plot_payoffs(True)

    print("####################")

if __name__ == '__main__':
    #main(4, prob_by_strategy_mod=prob_by_strategy_mod_4, k1=40, k2=5)
    #main(3, prob_by_strategy_mod=prob_by_strategy_mod_null, utils_costs='const', k1=30, k2=5)
    #main(4, prob_by_strategy_mod=prob_by_strategy_mod_4, utils_costs='exp', k1=40, k2=5)
    main(3, prob_by_strategy_mod=prob_by_strategy_mod_null, utils_costs='exp', k1=30, k2=5)