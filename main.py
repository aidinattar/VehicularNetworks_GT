from game import population
from utils_costs import *

def main():
    game = population(costs_exp, prob_by_strategy_mod_null, utils_const, 100, 100, 0.1, False)

    game.compute_payoffs()

    game.print_stats()
    game.print_payoffs()

    game.plot_strategies()

    print("####################")

if __name__ == '__main__':
    main()