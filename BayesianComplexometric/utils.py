import numpy as np 
from scipy.optimize import root, root_scalar, fsolve, brentq
from scipy import stats as st
from scipy.optimize import dual_annealing, minimize
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
			AL = AL / 1e9 # converting nano Molar to Molar
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
			AL = AL / 1e9 # converting nano Molar to Molar
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
				LT = X[:1]	
				K = 10**X[1:2]
				S = X[2:]
			else:
				LT = X[0:2]
				K = 10**X[2:4]
				S = X[4:]

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
				LT = X[0:1]	
				K = 10**X[1:2]

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
		return ret.x # optimized LT, K, and S
	

	if (S is None) and (AL is not None) and (KAL is not None):
		# ACSV method, so y_obs is the current signal equal to [MAL] * S
		def objFunc(X):
			if n_lig == 1:
				LT = X[0:1]	
				K = 10**X[1:2]
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
		return ret.x  # optimized LT, K, and S


	if (S is not None) and (AL is not None) and (KAL is not None):
		def objFunc(X):
	
			if n_lig == 1:
				LT = X[0:1]	
				K = 10**X[1:2]

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


def _resExp(y_obs, y_pred, relative_err = 0.03):
	# using analytical definition of normal distribution

	mu = y_pred - y_obs  
	
	cov = np.diag((y_obs * relative_err)**2)
	cov_inv = np.diag(1/np.diag(cov))  
	
	resExp = np.matmul(mu.reshape(1,-1), np.matmul(cov_inv, mu.reshape(-1,1))) # probability is proportional to exp(-0.5*resExp)
	
	return resExp

def _hyb_optim(y_obs,  MT, lb, ub, S, n_lig, AL, KAL):
	# this hybrid loss function optimization when S is unknown

	x = _optim(y_obs, MT, lb , ub, S, LossFunc = 'Gerringa', DeviationType = 'mse', n_lig = n_lig, AL = AL , KAL = KAL, optimizerKW = {'maxiter' : 100})
	x_ = _optim(y_obs, MT, lb[:-1] , ub[:-1], S = x[-1], LossFunc = 'Scatchard', DeviationType = 'mse', n_lig = n_lig, AL = AL , KAL = KAL, optimizerKW = {'maxiter' : 100})
	x[:-1] = x_

	return x


def _adjust_step(titr_model, x, S, y_obs, step, RR, PPR_target, relative_err, RR_opt = 0.766):

	# titr_model is the titration model that returns the observable from parameters in x when they are parsed into KLS
	step = np.array(step)
	step = step.copy()
	PPR = np.ones_like(x)   # this is only calculated in a single adjustment but PPR_target is modified from the last known
	if S is not None:
		S = S.copy()
	
	for j in range(x.size):
		x_ = x.copy()
		x_[j] = x[j].copy() + 0.5 * step[j]
		if S is None:
			LT_, K_, S_ = _x_toLKS(x)
			LT__, K__, S__ = _x_toLKS(x_)
		else:
			LT_, K_, _ = _x_toLKS(x)
			LT__, K__, _ = _x_toLKS(x_)
			S_ = S
			S__ = S
		y  = titr_model(LT_ , K_ , S_ )   # unperturbed
		y_ = titr_model(LT__, K__, S__)  # perturbed
		resExp = _resExp(y_obs, y, relative_err)
		resExp_ = _resExp(y_obs, y_, relative_err)
		PPR[j] = np.array([np.exp(-0.5 * (resExp.item() - resExp_.item()))])

	PPR[PPR>1] = 1 / PPR[PPR>1]
	PPR_target = PPR_target.copy() * (RR / RR_opt)  # scalar 
	if PPR_target < 1e-6:
		PPR_target = np.array(1e-6)

	PPR_diff = np.array(PPR_target) - np.array(PPR)

	jj = -1
	for ppr_d in PPR_diff:
		jj += 1
		if ppr_d > 0:
			step[jj] = step[jj] * np.exp((- ppr_d) /  PPR_target * 2)
		else:
			step[jj] = step[jj] * np.exp((- ppr_d) / (1 - PPR_target)*2) 

	return step, PPR_target



def _x_toLKS(x):
	# this function is used to parse and transform data for passing to the titration models
	x = np.array(x)
	if x.size == 2:
		LT = x[0:1].copy()
		K = 10**x[1:2].copy()
		S = None
	elif x.size == 3:
		LT = x[0:1].copy()
		K = 10**x[1:2].copy()
		S = x[2:].copy()
	elif x.size == 4:
		LT = x[:2].copy()
		K = 10**x[2:4].copy()
		S = None
	elif x.size == 5:
		LT = x[:2].copy()
		K = 10**x[2:4].copy()
		S = x[4:].copy()
	else:
		raise ValueError('this functions accept an array with 2 to 5 elements')
	return LT, K, S

def _new_point(x, step):

	step = np.array(step)
	if x.size != step.size:
		raise ValueError('the same number of steps must be known for the number of parameters')

	xp = x.copy() + step * st.uniform(-1,2).rvs(x.size)  

	return xp 


def _adjustOutOfRange(x, lb, ub):

	lb = np.array(lb)
	ub = np.array(ub)
	x = np.array(x)

	if x.size != lb.size != ub.size:
		raise ValueError('sizes of parameters and lower and upper bounds must be the same')

	# to make sure the original values won't be modified	
	tmp = x.copy()
	tmplb = lb.copy()
	tmpub = ub.copy()


	tmp[tmp > tmpub] = 2*tmpub[tmp > tmpub] - tmp[tmp > tmpub] 
	tmp[tmp < tmplb] = 2*tmplb[tmp < tmplb] - tmp[tmp < tmplb]

	return tmp


def _mcmc(x0, MT, y_obs, lb, ub, step0, relative_err, S = None, AL = None, KAL = None, niter = 60000):
	""" x0 ::  initial parameters; with log-transformed Ks
		MT ::  total metal concentration at titration points
		y_obs :: signal from experiment"""

	samples = np.zeros((niter, x0.size))
	samples[0,:] = x0
	NM = x0.size # number of parameters 
	PPR_target = np.array((1/2)**(1/NM))  # initial value; scalar; update after each step adjustment
	RR_opt = 0.766 # optimizal rejection rate

	naccept = 0

	if S is None:
		LT_0, K_0, S_0 = _x_toLKS(x0)	
	else:
		LT_0, K_0, _ = _x_toLKS(x0)	
		S_0 = S

	n_lig = LT_0.size


	if (S is None) and not (len(step0) == len(x0) == len(lb) == len(ub) == 2 * n_lig + 1):
		raise ValueError('This model requires {:d} value(s) for lower bound, upper bound, step0, and x0!'.format(2 * n_lig + 1))

	if (S is not None) and not (len(step0) == len(x0) == len(lb) == len(ub) == 2 * n_lig):
		raise ValueError('This model requires {:d} value(s) for lower bound, upper bound, step0, and x0!'.format(2 * n_lig))
	

	if  (AL is None) and (KAL is None):
		# method is ASV: M_free * S is the observed signal
		titr_model  = lambda LT, K, S : S * _titr_simulate(MT, LT, K, n_lig, AL, KAL)[:,0]
		for i in range(1, niter):

			if i==1:
				y_0 = titr_model(LT_0, K_0, S_0)

			if (i < 1000 and (i % 100) == 0) or (i >= 1000 and (i % 1000) == 0):
				RR = (i - naccept) / i
				step0, PPR_target =  _adjust_step(titr_model, x0, S, y_obs, step0, RR, PPR_target, relative_err)
				print(RR)
			# proposing a new point (x):
			xp = _new_point(x0, step0)
			# bouncing out-of-range back into the boundaries:
			xp = _adjustOutOfRange(xp, lb, ub)
			if S is None:	
				LT_p, K_p, S_p = _x_toLKS(xp)
			else:
				LT_p, K_p, _ = _x_toLKS(xp)
				S_p = S_0

			y_p = titr_model(LT_p, K_p, S_p)

			resExp_0 = _resExp(y_obs, y_0, relative_err)
			resExp_p = _resExp(y_obs, y_p, relative_err)
			rho = min(1, np.array([np.exp(-0.5 * (resExp_p.item() - resExp_0.item()))]))  # we grab item to make sure the result is scalar
			u = np.random.uniform()

			if u < rho:
				naccept += 1
				x0 = xp.copy()
				y_0 = y_p.copy()

			samples[i,:] = x0


	if  (AL is not None) and (KAL is not None):
		# method is ACSV: MAL * S is the observed signal
		titr_model  = lambda LT, K, S: S * _titr_simulate(MT, LT, K, n_lig, AL, KAL)[:,2]
		for i in range(1, niter):
			if i==1:
				y_0 = titr_model(LT_0, K_0, S_0)

			if (i < 1000 and (i % 100) == 0) or (i >= 1000 and (i % 1000) == 0):
				RR = (i - naccept) / i
				step0, PPR_target =  _adjust_step(titr_model, x0, S, y_obs, step0, RR, PPR_target, relative_err)
				print(RR)
			# proposing a new point (x)
			xp = _new_point(x0, step0)
			# bouncing out-of-range back into the boundaries
			xp = _adjustOutOfRange(xp, lb, ub)
			if S is None:	
				LT_p, K_p, S_p = _x_toLKS(xp)
			else:
				LT_p, K_p, _ = _x_toLKS(xp)
				S_p = S_0
			y_p = titr_model(LT_p, K_p, S_p)

			resExp_0 = _resExp(y_obs, y_0, relative_err)
			resExp_p = _resExp(y_obs, y_p, relative_err)
			rho = min(1, np.array([np.exp(-0.5 * (resExp_p.item() - resExp_0.item()))]))
			u = np.random.uniform()

			if u < rho:
				naccept += 1
				x0 = xp.copy()
				y_0 = y_p.copy()

			samples[i,:] = x0

	return samples

