import numpy as np 
from scipy.optimize import root, root_scalar, fsolve, brentq
from scipy.integrate import simps
from scipy.linalg import block_diag, det, eig
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import dual_annealing 
from functools import partial


from BayesianComplexometric.utils import _titr_simulate, _optim, _mcmc

LT = np.array([2, 8])
K = np.array([1e14, 1e12])
MT_ = np.linspace(0.1, 14, 12)

y_ = _titr_simulate(MT_, LT, K, n_lig = 2)[:,0]


y_obs = y_ + y_*0.03*np.random.normal(0, 1, MT_.size)
"""

plt.figure(figsize=[4 , 3])

plt.scatter(MT_, y_)
plt.scatter(MT_, y_obs)
plt.legend(['modeled','modeled + noise'])
plt.xlabel('total Metal [nM]')
plt.ylabel('free Metal [nM]')

plt.show()
"""
#x = _optim(y_obs * 0.6, MT = MT_, lb = [1, 5, 13, 11 , 0.1], ub = [4, 10, 15, 14, 1], S = None, LossFunc = 'Gerringa', DeviationType = 'mse', n_lig = 2, AL = None, KAL = None, \
#			optimizerKW = {'maxiter' : 2000})

#print(x)

x = np.array([1.9, 7.4, 13.8, 12.1,0.5])

samples = _mcmc(x , MT_, y_obs,  [1, 5, 13, 11 , 0.1], [4, 10, 15, 14, 1], [0.01 , 0.01, 0.002, 0.002, 0.002], 0.03, S = None, AL = None, KAL = None, niter = 60000)


"""

LT = np.array([2])
K = np.array([1e14])
MT_ = np.linspace(0.1,14,12) 

y_ = _titr_simulate(MT_, LT, K, n_lig = 1)

y_obs = y_ + y_*0.03*np.random.normal(0,1,MT_.size)

plt.figure(figsize=[4 , 3])

plt.scatter(MT_, y_)
plt.scatter(MT_,y_obs)
plt.legend(['modeled','modeled + noise'])
plt.xlabel('total Metal [nM]')
plt.ylabel('free Metal [nM]')


plt.show()

"""