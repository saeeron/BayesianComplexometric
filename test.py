import numpy as np 
from scipy.optimize import root, root_scalar, fsolve, brentq
from scipy.integrate import simps
from scipy.linalg import block_diag, det, eig
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import dual_annealing 
from functools import partial


from BayesianComplexometric.utils import _titr_simulate, _optim, _mcmc, _hyb_optim

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

lb =  [1, 5, 13, 11 , 0.1]
ub =  [4, 10, 15, 14, 1]

#lb =  [1, 11, 0.1]
#ub =  [10, 15, 1]
_hyb_optim(y_obs * 0.6,  MT = MT_, lb =lb, ub = ub, S = None, n_lig =2, AL = None, KAL= None)

x = _optim(y_obs * 0.6, MT = MT_, lb = lb, ub = ub, S = None, LossFunc = 'Gerringa', DeviationType = 'mse', n_lig = 2, AL = None, KAL = None, \
			optimizerKW = {'maxiter' : 100})


x2 = _optim(y_obs * 0.6, MT = MT_, lb = lb[:-1], ub = ub[:-1], S = x[-1], LossFunc = 'Scatchard', DeviationType = 'mse', n_lig = 2, AL = None, KAL = None, \
			optimizerKW = {'maxiter' : 100})



print(x)

#x = np.array([2, 8, 14, 12])

samples = _mcmc(x, MT_, y_obs * 0.6,  lb, ub, [0.01 , 0.01, 0.002, 0.002, 0.1], 0.03, S = None, AL = None, KAL = None, niter = 100000)


#samples = _mcmc(x , MT_, y_obs * 0.6,  [1, 5, 13, 11 , 0.1], [4, 10, 15, 14, 1], [0.01 , 0.01, 0.002, 0.002, 0.001], 0.03, S = None, AL = None, KAL = None, niter = 200000)

plt.subplot(5,1,1)
plt.plot(samples[20000:,0], 'g' , linewidth = 0.5)

plt.subplot(5,1,2)
plt.plot(samples[20000:,1], 'g' , linewidth = 0.5)


plt.hist(samples[20000:,3],100, color='g')

plt.show()



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