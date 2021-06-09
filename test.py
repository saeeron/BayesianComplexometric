import numpy as np 
import matplotlib.pyplot as plt



from BayesianComplexometric.utils import _titr_simulate, _optim, _mcmc, _hyb_optim

LT = np.array([2, 8])
K = np.array([1e14, 1e12])
MT_ = np.linspace(0.1, 14, 12)
S = 0.6

y_ = _titr_simulate(MT_, LT, K, n_lig = 2)[:,0]
y_obs = S*(y_ + y_*0.03*np.random.normal(0, 1, y_.size))


plt.figure(figsize=[4 , 3])
plt.scatter(MT_, y_obs, alpha = 0.4)
plt.legend(['Model + Noise'])
plt.xlabel('total Metal [nM]')
plt.ylabel('electric current (raw signal) ')

plt.show()


lb =  np.array([1, 5, 13, 11 , 0.1], dtype = 'float')
ub =  np.array([4, 10, 15, 14, 1], dtype = 'float')

x = _hyb_optim(y_obs,  MT = MT_, lb =lb, ub = ub, S = None, n_lig =2, AL = None, KAL= None)

print(x)

#x = np.array([2, 8, 14, 12])

samples = _mcmc(x, MT_, y_obs * 0.6,  lb, ub, [0.01 , 0.01, 0.002, 0.002, 0.1], relative_err = 0.03, S = None, AL = None, KAL = None, niter = 100000)


#samples = _mcmc(x , MT_, y_obs * 0.6,  [1, 5, 13, 11 , 0.1], [4, 10, 15, 14, 1], [0.01 , 0.01, 0.002, 0.002, 0.001], 0.03, S = None, AL = None, KAL = None, niter = 200000)


H, x_edge, y_edge = np.histogram2d(samples[20000:,3],samples[20000:,4],bins = 50, density=True)
H[H==0] = np.nan
plt.contourf(x_edge[:-1], y_edge[:-1], H.T, cmap="viridis")
plt.xlabel('K2')
plt.ylabel('S')
plt.show()



plt.subplot(5,1,1)
plt.plot(samples[20000:,0], 'g' , linewidth = 0.5)


plt.subplot(5,1,2)
plt.plot(samples[20000:,1], 'g' , linewidth = 0.5)


plt.hist(samples[20000:,4],100, color='g')

plt.show()

plt.scatter(samples[20000:,0], samples[20000:,4])

from pandas.plotting import scatter_matrix

df = pd.DataFrame(samples, columns = ['L1','L2','K1','K2','S'])

scatter_matrix(df, alpha=0.1, diagonal="kde")
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