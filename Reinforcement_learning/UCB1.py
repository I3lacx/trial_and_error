"""
Using confidence bounds as an algorithm strategy
Update:
X_mean = X_mean(j) + 2* ln(N)/N_j
X_mean(j) -> Old mean of Bandit j
N -> How many times did I tested everything
N_j -> How many times did I test this bandit j
It works because the more often we play everyone except for bandit j,
the higher the ration ln(N)/N_j will be
Be greedy all the time, similar to the optimistic_initial_value

So you update the means normally, but to find the best option, you use the
Upper confidence bound
"""