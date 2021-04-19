import numpy as np 
from scipy.optimize import root, root_scalar, fsolve, brentq
from scipy import stats as st
from scipy.optimize import dual_annealing 
from functools import partial

def _spec_forward_1L(MT, LT, K, AL = None, KAL = None ):
	# this fnction calclates M_free using a scalar root finding for 1-ligand case
	# LT is an array of [L1T]
	# K is an array of [K1]
	# S is the sensitivity Ip ~ M'
	if LT.size !=1 or K.size != 1:
		raise ValueError('Wrong model selectd!')

	LT = LT / 1e9 # converting nano Molar to Molar
	MT = MT / 1e9 # converting nano Molar to Molar

	if  ((AL is None) and (KAL is not None)) or ((AL is not None) and (KAL is None)):
		raise ValueError('If the Method is ACSV, both KAL and AL need to be specified!') 

	if (AL is None) and (KAL is None):
		# method is ASV
		def objf(X):
		# scalar function to be solved for the roots (X :: M_free) 
			ML1 = (K[0]*LT[0]*X/1e9) / (1+K[0]*X/1e9)
			return (MT - ML1 - X/1e9)

	else:
		# method is ACSV
		def objf(X):
		# scalar function to be solved for the roots (X :: M_free) 
			ML1 = (K[0]*LT[0]*X/1e9) / (1+K[0]*X/1e9)
			MAL = (KAL[0]*AL[0]*X/1e9) / (1+KAL[0]*X/1e9)
			return (MT - ML1 - MAL - X/1e9)


	initGuess = 10**(np.linspace(-12,3,20))
	sol = []
	for i in range(initGuess.size - 1):
		try:
			sol.append(brentq(objf, initGuess[i], initGuess[i+1], maxiter=100))
		except ValueError:
			sol.append(np.nan)
	M_free = np.nanmin(np.array(sol))
	all_M_L = MT - np.nanmin(np.array(sol))
	if (AL is not None) and (KAL is not None):
		MAL = KAL * M_free * AL / (1 + KAL * M_free)
	else:
		MAL = np.nan 

	return M_free


def _spec_forward_2L(MT, LT, K, AL = None, KAL = None):
	# this fnction calclates M_free using a scalar root finding for 2-ligand case
	# LT is an array of [L1T, L2T]
	# K is an array of [K1, K2]
	# S is the sensitivity Ip ~ M'

	if LT.size !=2 or K.size != 2:
		raise ValueError('Wrong model selectd!')

	LT = LT / 1e9 # converting nano Molar to Molar
	MT = MT / 1e9 # converting nano Molar to Molar

	if  ((AL is None) and (KAL is not None)) or ((AL is not None) and (KAL is None)):
		raise ValueError('If the Method is ACSV, both KAL and AL need to be specified!') 


	if (AL is None) and (KAL is None):
		# method is ASV
		def objf(X):
		# scalar function to be solved for the roots (M_free) 
			ML1 = (K[0]*LT[0]*X/1e9) / (1+K[0]*X/1e9)
			ML2 = (K[1]*LT[1]*X/1e9) / (1+K[1]*X/1e9)
			return (MT - ML1 - ML2 - X/1e9) 
	
	else:
		# method is ACSV
		def objf(X):
		# scalar function to be solved for the roots (X :: M_free) 
			ML1 = (K[0]*LT[0]*X/1e9) / (1+K[0]*X/1e9)
			ML2 = (K[1]*LT[1]*X/1e9) / (1+K[1]*X/1e9)
			MAL = (KAL[0]*AL[0]*X/1e9) / (1+KAL[0]*X/1e9)
			return (MT - ML1 - ML2 - MAL - X/1e9)


	initGuess = 10**(np.linspace(-12,3,20))
	sol = []
	for i in range(initGuess.size - 1):
		try:
			sol.append(brentq(objf, initGuess[i], initGuess[i+1], maxiter=100))
		except ValueError:
			sol.append(np.nan)
	
	M_free = np.nanmin(np.array(sol))
	all_M_L = MT - M_free
	if (AL is not None) and (KAL is not None):
		MAL = KAL * M_free * AL / (1 + KAL * M_free)
	else:
		MAL = np.nan 

	return M_free

def _titr_simulate(MT, LT, K, n_lig = 1, AL = None, KAL = None):
	""" it returns M_free for both ASV and ACSV """
	if n_lig == 1:
		return np.array([_spec_forward_1L(mt, LT, K, AL, KAL) for mt in MT])
	elif n_lig == 2:
		return np.array([_spec_forward_2L(mt, LT, K, AL, KAL) for mt in MT])
	else: 
		raise ValueError('accepted number of ligand classes: 1 or 2')


def _resExp(y_obs, MT, LT, K, relative_err = 0.03):
	# using analytical definition of normal distribution
	LT = X[:2]
	K = X[2:]
	y_ = _titr_simulate(MT, LT, K, n_lig = 1, AL = None, KAL = None)

	mu = y_ - y_obs  
	  
	cov = np.diag((y_obs * relative_err)**2)
	cov_inv = np.diag(1/np.diag(cov)) # 
	  
	resExp = np.matmul(mu.reshape(1,-1), np.matmul(cov_inv, mu.reshape(-1,1)))
	  
	#return  ((2*np.pi)**(- y_obs.size /2)) * (det(cov) ** -0.5) * np.exp(-0.5 * resExp)
	return  resExp



def _optim(y_obs, MT, lb, ub, LossFunc = 'Mfree', DeviationType = 'mse', n_lig = 1, AL = None, KAL = None, S = 1 ):
	""" Single-point optimization based on a variety of loss functions (vdB/Ruzic, Scatchard)  """
	if lb.size != n_lig or ub.size !=n_lig:
		raise ValueError('This model requires {:d} value(s) for lower bound and upper bound!'.format(n_lig))
	
	if DeviationType not in ['mse','mae']:
		raise ValueError('Choose Deviation Type to be one of \"mse\" or \"mae\"!')
	
	if LossFunc not in ['Mfree', 'Scatchard' , 'vdB']:
		raise ValueError('Choose Deviation Type to be one of \"Mfree\", \"Scatchard\", or \"vdB\"!')
	

	def objFunc(X):
		if n_lig == 1:
			LT = X[0]
			K = X[1]
		else:
			LT = X[:2]
			K = X[2:]

		y_pred = S * _titr_simulate(MT, LT, K, n_lig, AL, KAL)
		if LossFunc == 'Mfree':
				
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((y_pred - y_obs)**2))
				else:
					floss = np.mean(np.abs(y_pred - y_obs))

		if LossFunc == 'Scatchard':
				
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((y_pred - y_obs)**2))
				else:
					floss = np.mean(np.abs(y_pred - y_obs))





def _adjust_step():
	pass



