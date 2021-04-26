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
	AL = AL / 1e9

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
	
	M_free = np.nanmin(np.array(sol))  # nano molar
	
	MT = MT * 1e9  # converting back to nano molar

	if (AL is not None) and (KAL is not None):
		MAL = 1e9 * (KAL * M_free/1e9 * AL/1e9 / (1 + KAL * M_free/1e9)) # nano molar
		all_M_L = MT - M_free - MAL  # nano molar
	else:
		MAL = 0
		all_M_L = MT - M_free 

	return [M_free, all_M_L, MAL]


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

	M_free = np.nanmin(np.array(sol))  # nano molar
	
	MT = MT * 1e9  # converting back to nano molar

	if (AL is not None) and (KAL is not None):
		MAL = 1e9 * (KAL * M_free/1e9 * AL/1e9 / (1 + KAL * M_free/1e9)) # nano molar
		all_M_L = MT - M_free - MAL  # nano molar
	else:
		MAL = 0
		all_M_L = MT - M_free 

	return [M_free, all_M_L, MAL]


def _titr_simulate(MT, LT, K, n_lig = 1, AL = None, KAL = None):
	""" it returns M_free for both ASV and ACSV """

	if  ((AL is None) and (KAL is not None)) or ((AL is not None) and (KAL is None)):
		raise ValueError('If the Method is ACSV, both KAL and AL need to be specified!') 

	if n_lig == 1:
		return np.array([_spec_forward_1L(mt, LT, K, AL, KAL) for mt in MT]) # columns : Mfree, all MLs, and MAL

	elif n_lig == 2:
		return np.array([_spec_forward_2L(mt, LT, K, AL, KAL) for mt in MT]) # columns : Mfree, all MLs, and MAL
		
	else: 	
		raise ValueError('accepted number of ligand classes: 1 or 2')


def _resExp(y_obs, MT, LT, K, relative_err = 0.03):
	# using analytical definition of normal distribution
	LT = X[:2]
	K = X[2:]
	y_ = _titr_simulate(MT, LT, K, n_lig = 1, AL = None, KAL = None)

	mu = y_ - y_obs  
	  
	cov = np.diag((y_obs * relative_err)**2)
	cov_inv = np.diag(1/np.diag(cov))  
	  
	resExp = np.matmul(mu.reshape(1,-1), np.matmul(cov_inv, mu.reshape(-1,1))) # probalility is proportional to exp(-0.5*resExp)
	  
	
	return resExp


def _resExpInitM( MT_, MT, relative_err = 0.03):
	# MT_ and MT are two initial metal concetrations. MT or true metal concentration whose pdf is to be known

	mu = MT - MT0
	cov = (MT * relative_err)**2
	cov_inv = 1/cov  

	resExp =  mu * cov_inv * mu  # probalility is proportional to exp(-0.5*resExp)

	return resExp

def _optim(y_obs, MT, lb, ub, S = None, LossFunc = 'Mfree', DeviationType = 'mse', n_lig = 1, AL = None, KAL = None, optimizerKW = {} ):
	""" Single-point optimization based on a variety of loss functions (vdB/Ruzic, Scatchard)  """
	if 'maxiter' in optimizerKW:
		maxiter = optimizerKW['maxiter']

	lb = np.array(lb) 
	ub = np.array(ub) 

	if (S is None) and (len(lb) != 2* n_lig + 1 or len(ub) != 2 * n_lig + 1):
		raise ValueError('This model requires {:d} value(s) for lower bound and upper bound!'.format(2 * n_lig + 1))

	if (S is not None) and (len(lb) != 2* n_lig or len(ub) != 2 * n_lig):
		raise ValueError('This model requires {:d} value(s) for lower bound and upper bound!'.format(2 * n_lig))
	
	if DeviationType not in ['mse','mae']:
		raise ValueError('Choose Deviation Type to be one of \"mse\" or \"mae\"!')
	
	if LossFunc not in ['Mfree', 'Scatchard' , 'vdB', 'Gerringa']:
		raise ValueError('Choose Deviation Type to be one of \"Mfree\", \"Scatchard\", \"vdB\", or \"Gerringa\"!')


	# defining different loss functions 
	
	if (S is None) and (AL is None) and (KAL is None):
		#ASV 
		# we need to find S as well as ligand concentration and constant
		def objFunc(X):
			if n_lig == 1:
				LT = X[0]	
				K = 10**X[1]
				S = X[2]
			else:
				LT = X[0:2]
				K = 10**X[2:4]
				S = X[4]

			tmp = _titr_simulate(MT, LT, K, n_lig, AL, KAL ) 

			if LossFunc == 'Mfree':
				Mfree_obs = y_obs / S
				Mfree_pred = tmp[:,0]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((Mfree_pred - Mfree_obs)**2))
				else:
					floss = np.mean(np.abs(Mfree_pred - Mfree_obs))
			
			if LossFunc == 'Scatchard':
				Mfree_obs = y_obs / S
				all_M_L_obs = MT - Mfree_obs 
				allML_Mfree_obs = all_M_L_obs / Mfree_obs
				allML_Mfree_pred = tmp[:,1] / tmp[:,0] 
				if DeviationType == 'mse': 
					floss = np.sqrt(np.mean((allML_Mfree_pred - allML_Mfree_obs)**2))
				else:
					floss = np.mean(np.abs(allML_Mfree_pred - allML_Mfree_obs))

			if LossFunc == 'vdB':
				Mfree_obs = y_obs / S
				all_M_L_obs = MT - Mfree_obs
				Mfree_allML_obs = Mfree_obs / all_M_L_obs 
				Mfree_allML_pred = tmp[:,0] / tmp[:,1]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((Mfree_allML_obs - Mfree_allML_pred)**2))
				else:
					floss = np.mean(np.abs(Mfree_allML_obs - Mfree_allML_pred))

			if LossFunc == 'Gerringa':
				Mfree_obs = y_obs / S
				all_M_L_obs = MT - Mfree_obs				
				all_M_L_pred = tmp[:,1]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((all_M_L_obs - all_M_L_pred)**2))
				else:
					floss = np.mean(np.abs(all_M_L_obs - all_M_L_pred))

			return floss
		
		ret = dual_annealing(func = objFunc, bounds = list(zip(lb,ub)), maxiter = maxiter, seed=2442)
		return ret.x # optimized LT, K, and S

	if (S is not None) and (AL is None) and (KAL is None):
		#ASV
		def objFunc(X):
			if n_lig == 1:
				LT = X[0]	
				K = 10**X[1]

			else:
				LT = X[0:2]
				K = 10**X[2:4]
				
			tmp = _titr_simulate(MT, LT, K, n_lig, AL, KAL ) 

			if LossFunc == 'Mfree':
				Mfree_obs = y_obs / S
				Mfree_pred = tmp[:,0]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((Mfree_pred - Mfree_obs)**2))
				else:
					floss = np.mean(np.abs(Mfree_pred - Mfree_obs))
			
			if LossFunc == 'Scatchard':
				Mfree_obs = y_obs / S
				all_M_L_obs = MT - Mfree_obs 
				allML_Mfree_obs = all_M_L_obs / Mfree_obs
				allML_Mfree_pred = tmp[:,1] / tmp[:,0] 
				if DeviationType == 'mse': 
					floss = np.sqrt(np.mean((allML_Mfree_pred - allML_Mfree_obs)**2))
				else:
					floss = np.mean(np.abs(allML_Mfree_pred - allML_Mfree_obs))

			if LossFunc == 'vdB':
				Mfree_obs = y_obs / S
				all_M_L_obs = MT - Mfree_obs
				Mfree_allML_obs = Mfree_obs / all_M_L_obs 
				Mfree_allML_pred = tmp[:,0] / tmp[:,1]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((Mfree_allML_obs - Mfree_allML_pred)**2))
				else:
					floss = np.mean(np.abs(Mfree_allML_obs - Mfree_allML_pred))

			if LossFunc == 'Gerringa':
				Mfree_obs = y_obs / S
				all_M_L_obs = MT - Mfree_obs				
				all_M_L_pred = tmp[:,1]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((all_M_L_obs - all_M_L_pred)**2))
				else:
					floss = np.mean(np.abs(all_M_L_obs - all_M_L_pred))

			return floss

		ret = dual_annealing(func = objFunc, bounds = list(zip(lb,ub)), maxiter = maxiter, seed=2442)
		return ret.x  # optimized LT, and K

	if (S is None) and (AL is not None) and (KAL is not None):
		# ACSV method, so y_obs is the current signal equal to [MAL] * S
		def objFunc(X):
			if n_lig == 1:
				LT = X[0]	
				K = 10**X[1]
				S = X[2]
			else:
				LT = X[0:2]
				K = 10**X[2:4]
				S = X[4]
		
			tmp = _titr_simulate(MT, LT, K, n_lig, AL, KAL ) 

			if LossFunc == 'Mfree':
				MAL_obs = y_obs / S  # nM
				Mfree_obs = MAL_obs /1e9 / KAL / (AL/1e9 - MAL_obs/1e9) * 1e9 # nM
				Mfree_pred = tmp[:,0]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((Mfree_pred - Mfree_obs)**2))
				else:
					floss = np.mean(np.abs(Mfree_pred - Mfree_obs))
			
			if LossFunc == 'Scatchard':
				MAL_obs = y_obs / S  # nM
				Mfree_obs = MAL_obs /1e9 / KAL / (AL/1e9 - MAL_obs/1e9) * 1e9 # nM
				all_M_L_obs = MT - Mfree_obs - MAL_obs 
				allML_Mfree_obs = all_M_L_obs / Mfree_obs
				allML_Mfree_pred = tmp[:,1] / tmp[:,0] 
				if DeviationType == 'mse': 
					floss = np.sqrt(np.mean((allML_Mfree_pred - allML_Mfree_obs)**2))
				else:
					floss = np.mean(np.abs(allML_Mfree_pred - allML_Mfree_obs))

			if LossFunc == 'vdB':
				MAL_obs = y_obs / S
				Mfree_obs = MAL_obs /1e9 / KAL / (AL/1e9 - MAL_obs/1e9) * 1e9 # nM
				all_M_L_obs = MT - Mfree_obs - MAL_obs 
				Mfree_allML_obs = Mfree_obs / all_M_L_obs 
				Mfree_allML_pred = tmp[:,0] / tmp[:,1]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((Mfree_allML_obs - Mfree_allML_pred)**2))
				else:
					floss = np.mean(np.abs(Mfree_allML_obs - Mfree_allML_pred))

			if LossFunc == 'Gerringa':
				MAL_obs = y_obs / S
				Mfree_obs = MAL_obs /1e9 / KAL / (AL/1e9 - MAL_obs/1e9) * 1e9 # nM
				all_M_L_obs = MT - Mfree_obs - MAL_obs				
				all_M_L_pred = tmp[:,1]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((all_M_L_obs - all_M_L_pred)**2))
				else:
					floss = np.mean(np.abs(all_M_L_obs - all_M_L_pred))


			return floss

		ret = dual_annealing(func = objFunc, bounds = list(zip(lb,ub)), maxiter = maxiter, seed=2442)
		return ret.x   # optimized LT, K, and S


	if (S is not None) and (AL is not None) and (KAL is not None):
		def objFunc(X):
	
			if n_lig == 1:
				LT = X[0]	
				K = 10**X[1]

			else:
				LT = X[0:2]
				K = 10**X[2:4]
		
			tmp = _titr_simulate(MT, LT, K, n_lig, AL, KAL ) 

			if LossFunc == 'Mfree':
				MAL_obs = y_obs / S  # nM
				Mfree_obs = MAL_obs /1e9 / KAL / (AL/1e9 - MAL_obs/1e9) * 1e9 # nM
				Mfree_pred = tmp[:,0]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((Mfree_pred - Mfree_obs)**2))
				else:
					floss = np.mean(np.abs(Mfree_pred - Mfree_obs))
			
			if LossFunc == 'Scatchard':
				MAL_obs = y_obs / S  # nM
				Mfree_obs = MAL_obs /1e9 / KAL / (AL/1e9 - MAL_obs/1e9) * 1e9 # nM
				all_M_L_obs = MT - Mfree_obs - MAL_obs 
				allML_Mfree_obs = all_M_L_obs / Mfree_obs
				allML_Mfree_pred = tmp[:,1] / tmp[:,0] 
				if DeviationType == 'mse': 
					floss = np.sqrt(np.mean((allML_Mfree_pred - allML_Mfree_obs)**2))
				else:
					floss = np.mean(np.abs(allML_Mfree_pred - allML_Mfree_obs))

			if LossFunc == 'vdB':
				MAL_obs = y_obs / S
				Mfree_obs = MAL_obs /1e9 / KAL / (AL/1e9 - MAL_obs/1e9) * 1e9 # nM
				all_M_L_obs = MT - Mfree_obs - MAL_obs 
				Mfree_allML_obs = Mfree_obs / all_M_L_obs 
				Mfree_allML_pred = tmp[:,0] / tmp[:,1]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((Mfree_allML_obs - Mfree_allML_pred)**2))
				else:
					floss = np.mean(np.abs(Mfree_allML_obs - Mfree_allML_pred))
			
			if LossFunc == 'Gerringa':
				MAL_obs = y_obs / S
				Mfree_obs = MAL_obs /1e9 / KAL / (AL/1e9 - MAL_obs/1e9) * 1e9 # nM
				all_M_L_obs = MT - Mfree_obs - MAL_obs				
				all_M_L_pred = tmp[:,1]
				if DeviationType == 'mse':
					floss = np.sqrt(np.mean((all_M_L_obs - all_M_L_pred)**2))
				else:
					floss = np.mean(np.abs(all_M_L_obs - all_M_L_pred))

			return floss


		ret = dual_annealing(func = objFunc, bounds = list(zip(lb,ub)), maxiter = maxiter, seed=2442)
		return ret.x   # optimized LT, and K




def _adjust_step():
	pass



