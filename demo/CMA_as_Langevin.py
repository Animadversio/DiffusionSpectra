import numpy as np
import torch
#%%
# if population_size is None:
#     self.lambda_ = int(4 + floor(3 * log2(N)))  # population size, offspring number
#     # the relation between dimension and population size.
# else:
#     self.lambda_ = population_size  # use custom specified population size
population_size = 30
mu = population_size / 2  # number of parents/points for recombination
#  Select half the population size as parents
weights = np.log(mu + 1 / 2) - (np.log(np.arange(1, 1 + np.floor(mu))))
weights = weights / weights.sum()
weights_full = np.concatenate((weights, np.zeros(int(mu))))
#%%
reps = 10000
vectors = np.random.randn(population_size, reps)
nograd_result = weights_full@vectors
vectors = np.random.randn(population_size, reps)
vectors_sort = np.sort(vectors, axis=0)
grad_result = weights_full@vectors_sort[::-1,:]
#%
import matplotlib.pyplot as plt
plt.figure()
plt.hist(grad_result, alpha=0.5, bins=100)
plt.hist(nograd_result, alpha=0.5, bins=100)
plt.title(f"grad {grad_result.mean():.5f}+-{grad_result.std():.5f}\n"
          f"nograd {nograd_result.mean():.5f}+-{nograd_result.std():.5f}\n")
plt.show()
#%%
# for analysis of CMA ES there is essentially one key parameter which is
# population size, the statistical property of the updates depend on the
# one parameter
