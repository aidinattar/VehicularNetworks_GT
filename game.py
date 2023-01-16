'''
Simple Repeated game

Trying to implement a simple repeated game where for each time step the payoffs
and costs are updated. The players do not consider the future, so that at each
time step we can simply perform a normal game and search for NE.

In this game we assume each player can choose from 4 strategies, each more
powerful than the previous one in attacking/defending.


'''

import nashpy as nash
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm

np.random.seed(1)

class game(object):
    '''
    Class for the repeated game
    '''

    def __init__(self, costs, prob_by_strategy_mod, utils,
                 n_steps=1000, L=.01, debug=False):
        '''
        Initialize the game

        Parameters:
        -----------
        costs: dict
            costs for each strategy
        prob_by_strategy_mod: np.array
            probability of winning for the attacker for each strategy
        utils: dict
            utilities for each strategy
        n_steps: int
            number of time steps
        L: float
            learning rate
        debug: bool
            debug flag
        '''
        self.costs = costs
        self.prob_by_strategy_mod = prob_by_strategy_mod
        self.utils = utils
        self.n_steps = n_steps
        self.L = L

        self.n_strategies = prob_by_strategy_mod.shape[0]
        self.debug = debug


    def get_payoffs(self, t):
        ######### get the costs and utilities #########
        row_costs = self.costs['row']
        column_costs = self.costs['column']

        row_utils = self.utils['row']
        column_utils = self.utils['column']

        ######### get the probabilities #########
        prob_by_strategy_mod = self.prob_by_strategy_mod


        ## sample the probaility of succeeding attack ##
        prob = np.exp(-self.L*t)

        ######### obtain succeed or fail mask #########
        sample = random()
        mask   = (prob*prob_by_strategy_mod) > sample

        return np.where(mask, row_utils[0,:,:], row_utils[1,:,:]) - np.reshape(row_costs, (-1,1)), \
               np.where(mask, column_utils[1,:,:], column_utils[0,:,:]) - column_costs # attention to numpy broadcasting


    def run(self):
        '''
        Run the game
        '''
        degeneracies = []
        strategies   = np.empty((self.n_steps, 2, self.n_strategies))
        payoffs      = np.empty((self.n_steps, 2))

        #for i in tqdm(range(self.n_steps), 'running game'):
        for i in range(self.n_steps):

            mygame = nash.Game(*self.get_payoffs(i))
            tmp = np.stack(list(mygame.support_enumeration()))
            if(tmp.shape[0] > 1):
                if(self.debug):
                    print(f'Got a degenerate game at iteration {i}')
                degeneracies.append((i, mygame))

            strategies[i, :, :] = tmp[0, :, :]

            payoffs[i,:] = mygame[tmp[0, :, :]]

        self.strategies = strategies
        self.payoffs    = payoffs
        self.degeneracies = degeneracies


    def plot_strategies_cum(self):
        '''
        Plot the strategies of the population
        '''

        # plot the number of times a strategy has been played as a function of the number of matches played
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True, sharey=True)
        for i in range(2):
            axes[i].plot(np.array(range(self.n_steps)), self.strategies[0,:, i, :].cumsum(axis=0))
            #axes[i].legend([f"$\sigma_{0}$", f"$\sigma_{1}$", f"$\sigma_{2}$"])
            axes[i].legend([f"$\sigma_{j}$" for j in range(self.n_strategies)])
            axes[i].grid(axis='y', alpha=0.5)
            axes[i].set_title(f'Player {i+1}')

        fig.suptitle('Cumsum of played strategies in first game')
        plt.show()


class population(game):
    '''
    Class for the repeated game
    '''

    def __init__(self, costs, prob_by_strategy_mod, utils,
                 population_size=100, n_steps=1000, L=.01, debug=False):
        '''
        Initialize the game

        Parameters:
        -----------
        costs: dict
            costs for each strategy
        prob_by_strategy_mod: np.array
            probability of winning for the attacker for each strategy
        utils: dict
            utilities for each strategy
        population_size: int
            number of players in the population
        n_steps: int
            number of time steps
        L: float
            learning rate
        debug: bool
            debug flag
        '''

        super().__init__(costs, prob_by_strategy_mod, utils, n_steps, L, debug)

        self.population_size = population_size

    def compute_payoffs(self):
        '''
        Compute the payoffs for the population
        '''
        strategies = []
        payoffs = []
        degeneracies = []

        for i in tqdm(range(self.population_size), 'population'):
            tmp = game(self.costs, self.prob_by_strategy_mod, self.utils, self.n_steps, self.L, self.debug)
            tmp.run()
            strategies.append(tmp.strategies)
            payoffs.append(tmp.payoffs)
            degeneracies.append(tmp.degeneracies)

        self.strategies   = np.stack(strategies)
        self.payoffs      = np.stack(payoffs)
        self.degeneracies = degeneracies


    def print_stats(self):
        '''
        Print some stats
        '''
        print(f'Population size: {self.population_size}')
        print(f'Number of time steps: {self.n_steps}')
        print(f'Learning rate: {self.L}')
        print(f'Number of degenerate games: {len(self.degeneracies)}')

    def print_payoffs(self):
        '''
        Print the payoffs
        '''
        print(f'Average payoffs:\n\trow: {self.payoffs.sum(axis=1).mean(axis=0)[0]} \n\tcolumn: {self.payoffs.sum(axis=1).mean(axis=0)[1]}')

    def print_degeneracies(self, n=10):
        '''
        Print the degenerate games

        Parameters:
        -----------
        n: int
            number of degenerate games to print
        '''
        print(f'Degenerate games: {self.degeneracies[n]}')

    def degenerate_games_inspection(self, i=0, j=0, k=1):
        '''
        Inspect the degenerate games

        Parameters:
        -----------
        i: int
            index of the degenerate game
        j: int
            index of the player
        k: int
            index of the strategy
        '''

        deg_game = self.degeneracies[i][j][k]
        eqs = np.stack(list(deg_game.support_enumeration()))
        print('Payoffs of degenerate game:')
        for l, s in enumerate(eqs):
            print(f'Equilibrium {l}:', deg_game[s])

    def plot_strategies(self):
        '''
        Plot the strategies of the population
        '''

        # prob. density of strategies in function of game iteration based on population
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True, sharey=True)
        for i in range(2):
            axes[i].plot(np.array(range(self.n_steps)), self.strategies[:, :, i, :].mean(axis=0))
            #axes[i].legend([f"$\sigma_{0}$", f"$\sigma_{1}$", f"$\sigma_{2}$"])
            axes[i].legend([f"$\sigma_{j}$" for j in range(self.n_strategies)])
            axes[i].grid(axis='y', alpha=0.5)
            axes[i].set_title(f'Player {i+1}')

        fig.suptitle('PDF of strategies')
        plt.show()


    def plot_payoffs(self):
        '''
        Plot the payoffs of the population
        '''

        # evolution of payoff as a function of number of played matches
        plt.plot(np.array(range(self.n_steps)), self.payoffs.cumsum(axis=1).mean(axis=0))
        plt.legend(["Player 1", "Player2"])
        plt.show()