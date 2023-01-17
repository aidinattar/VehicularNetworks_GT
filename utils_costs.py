import numpy as np

costs_simple = {
    'row'    : np.array([1, 5, 10]),
    'column' : np.array([1, 5, 20])
}

prob_by_strategy_mod_simple = np.array([[1, 2, 4],
                                        [0.5, 1, 2],
                                        [0.25, 0.5, 1]])

utils_simple = {
    'row' : np.array([[[-10, -20, -50],
                       [-5, -10, -20],
                       [-1, -5, -10]],
                      [[2, 0, 0],
                       [5, 2, 0],
                       [10, 5, 2]]]),
    'column' : np.array([[[-5, -1, 0],
                          [-10, -5, -1],
                          [-20, -10, -5]],
                         [[10, 20, 50],
                          [5, 10, 20],
                          [1, 5, 10]]])
}


c = [1, 2]
n_strategies = 3
costs_exp = {
    'row'   : np.fromiter((c[0] * (2 ** (2 * n) - 1) for n in range(n_strategies)), dtype=float),
    'column': np.fromiter((c[1] * (2 ** (2 * n) - 1) for n in range(n_strategies)), dtype=float)
}


row_value_lose = -20
row_value_win  = 0
column_value_lose = 0
column_value_win  = 20
utils_const = {
    'row'   : np.array([np.full((n_strategies, n_strategies), row_value_lose),    np.full((n_strategies, n_strategies), row_value_win)]),
    'column': np.array([np.full((n_strategies, n_strategies), column_value_lose), np.full((n_strategies, n_strategies), column_value_win)])
}


prob_by_strategy_mod_null = np.array([[0, np.inf, np.inf],
                                        [0, 1, 2],
                                        [0, 0.5, 1]])

utils_simple_null = {
    'row' : np.array([[[0, -20, -50],
                       [np.nan, -10, -20],
                       [np.nan, -5, -10]],
                      [[0, np.nan, np.nan],
                       [5, 2, 0],
                       [10, 5, 2]]]),
    'column' : np.array([[[0, np.nan, np.nan],
                          [-10, -5, -1],
                          [-20, -10, -5]],
                         [[0, 20, 50],
                          [np.nan, 10, 20],
                          [np.nan, 5, 10]]])
}

utils_4_null = {
    'row' : np.array([[[0, -10, -20, -50],
                       [np.nan, -5, -10, -20],
                       [np.nan, -2, -5, -10],
                       [np.nan, -1, -2, -5]],
                      [[0, np.nan, np.nan, np.nan],
                       [5, 2, 1, 0],
                       [10, 5, 2, 1],
                       [20, 10, 5, 2]]]),
    'column' : np.array([[[0, np.nan, np.nan, np.nan],
                          [-10, -5, -2, -1],
                          [-20, -10, -5, -2],
                          [-50, -20, -10, -5]],
                         [[0, 10, 20, 50],
                          [np.nan, 5, 10, 20],
                          [np.nan, 2, 5, 10],
                          [np.nan, 1, 2, 5]]])
}

prob_by_strategy_mod_4 = np.array([[0, np.inf, np.inf, np.inf],
                                   [0, 1, 2, 4],
                                   [0, 0.5, 1, 2],
                                   [0, 0.25, 0.5, 1]])


c = [1, 2]
n_strategies = 4
costs_exp_4 = {
    'row'   : np.fromiter((c[0] * (2 ** (2 * n) - 1) for n in range(n_strategies)), dtype=float),
    'column': np.fromiter((c[1] * (2 ** (2 * n) - 1) for n in range(n_strategies)), dtype=float)
}