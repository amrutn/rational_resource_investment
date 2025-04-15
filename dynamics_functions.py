import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from scipy.special import expit # sigmoid function

# rng for this code
rng = np.random.default_rng(42)

def gen_dist(sigma, beta, N=100):
	"""
	Generate P(y|x) for each grid point, x with varying volatility
	over x and uncertainty.

	Params
	------
	sigma : float
		Standard deviation of the Gaussian smoothing filter, 
		controls volatility. Larger sigma = smoother field.
	beta : float
		Gain of the sigmoid nonlinearity, controls uncertainty.
		Larger gain = probabilities closer to 0 or 1.
	N : int
		Size of the grid on each side
	Returns
	-------
	dist : np array (N x N x 2)
		Generated conditional distribution P(y|x). 
		dist[i,j,0] = P(0|x_ij)
		dist[i,j,1] = P(1|x_ij)
		The marginalized P(y=1) is approximately 0.5
	"""
	if sigma < 0:
		raise ValueError("sigma must be non-negative")
	if beta < 0:
		raise ValueError("beta must be non-negative")

	# Generate base noise over grid
	base = rng.standard_normal(size=(N,N))

	# Apply Gaussian filter
	if sigma > 1e-6:
		field = gaussian_filter(base, sigma=sigma)
	else:
		field = base # no smoothing if sigma is near 0

	# Normalize for 0 mean, std 1
	field -= np.mean(field)
	std = np.std(field)
	if std > 1e-9:
		field /=std # avoid dividing by 0 if field is constant

	# map to probabilities
	p_y1 = expit(beta * field)

	# final dist:
	dist = np.zeros((N, N, 2))
    dist[:, :, 1] = p_y1       # P(y=1|x)
    dist[:, :, 0] = 1 - p_y1   # P(y=0|x)

    return dist

def gen_samples(num_samples, dist):
	"""
	Generate random samples according to dist=P(y|x) and uniformly
	distributed over x.

	Params
	------
	num_samples : int
		Number of samples to generate.
	dist : np array (N x N x 2)
		Conditional P(y|x), output of gen_dist.
		
	Returns
	-------
	samples : np array (num_samples, 3)
		Array of sample (coordinates, label)
	"""
	N = dist.shape[0]
	# Generate num_samples random coordinates
	coords = rng.integers(0,high=N,size=(num_samples,2))

	# Probability of y=1 for each of these coordinates
	P_y1 = dist[coords[:,0],coords[:,1]]

	# Stochastically assign labels 1 or 0
	rand_nums = rng.uniform(size=num_samples)
	labels = rand_nums < P_y1

	return np.hstack((R,labels[:,None]))


# Compute g_1'/P^*(x)
def g1_prime(p):
	"""
	Computes derivative of the mutual information constraint
	with respect to P(y|x)

	Params
	------
	p : 3D numpy array (N x N x 2)
		P(y|x) where
		p[i,j,0] = P(0|x_ij)
		p[i,j,1] = P(1|x_ij)

	Returns
	-------
	deriv : 3D numpy array (N x N x 2)
		Derivative of Mutual Information
	"""
	N = p.shape[0]
	# Compute marginalized distribution P(y)
	p_margin = p.mean(axis=(0,1), keepdims=True)
	

	return np.log(p) - np.log(p_margin)

# Compute \tilde g_1/P^*(x)
def g1_tilde(p):
	"""
	Computes mean-subtracted g1_prime.

	Params
	------
	p : 3D numpy array (N x N x 2)
		P(y|x) where
		p[i,j,0] = P(0|x_ij)
		p[i,j,1] = P(1|x_ij)

	Returns
	-------
	mean_sub_deriv : 3D numpy array (N x N x 2)
		Derivative of Mutual Information with 0 mean
	"""
	deriv = g1_prime(p)
	deriv_mean = (deriv*p).sum(axis=-1,keepdims=True)

	return deriv - deriv_mean

# Compute g2'/P^*(x) after setting \delta x = 1
def g2_prime(p):
	"""
	Computes derivative of the smoothness constraint
	with respect to P(y|x)

	Params
	------
	p : 3D numpy array (N x N x 2)
		P(y|x) where
		p[i,j,0] = P(0|x_ij)
		p[i,j,1] = P(1|x_ij)

	Returns
	-------
	deriv : np array (N x N x 2)
		Derivative of the distribution volatility
	"""
	N = p.shape[0]
	p1 = np.roll(p,1,axis=0)
	p1[0,:] = p[0,:] # Boundary condition

	p2 = np.roll(p,1,axis=1)
	p2[:,0] = p[:,0] # Boundary condition

	p3 = np.roll(p,-1,axis=0)
	p3[-1,:] = p[-1,:] # Boundary condition

	p4 = np.roll(p,-1,axis=1)
	p4[:,-1] = p[:,-1] # Boundary condition

	return  ((4*p - p1 - p2 - p3 - p4)/p 
		+ (4*np.log(p)-np.log(p1)-np.log(p2)-np.log(p3)-np.log(p4)))

# Compute \tilde g_2/P^*(x)
def g2_tilde(p):
	"""
	Computes mean-subtracted g2_prime.

	Params
	------
	p : 3D numpy array (N x N x 2)
		P(y|x) where
		p[i,j,0] = P(0|x_ij)
		p[i,j,1] = P(1|x_ij)

	Returns
	-------
	mean_sub_deriv : 3D numpy array (N x N x 2)
		Derivative of volatility with 0 mean
	"""
	deriv = g2_prime(p)
	deriv_mean = (deriv*p).sum(axis=-1,keepdims=True)

	return deriv - deriv_mean

def get_dist(samples, a, b, c, p0=None, N=100, T=1000):
	"""
	Compute learning dynamics by integrating first-order ODE
	equation of motion. Recall, we set delta x = 0 

	Params
	------
	samples : np array (num_samples, 3)
		Array of sample coordinates, label as generated by gen_samples
	a : float
		Parameter controlling influence of new data samples.
	b : float
		Parameter controlling allowed distribution volatility.
	c : float
		Parameter controlling expected mutual information.
	p0 : np array (N x N x 2)
		Initial conditional distribution. 
		If None, will default to uniform distribution.
	N : int
		number of bins in each axis of the internal representation
	T : float
		amount of time to run integration
	
	Returns
	-------
	t : np array (T)
		List of timepoints solution was evaluated at
	p : np array (T x N x N x 2)
		Learning trajectory distribution
	Message : str
		Cause of ODEInt termination
	"""
	T=int(T) # we only save output at integer timesteps

	if p0 is None:
		p0 = .5*np.ones(2*N**2) # uniform init
	else:
		assert N == p0.shape[0]
		p0 = p0.reshape(2*N**2)

	# Times to save the probabilities
	t_eval = np.linspace(0, T, T)


	# Compute time derivative of p
	def fun(t,p):

		p = p.reshape((N,N,2))

		# current source term
		m = np.zeros((N,N,2))
		m[tuple(samples[int(t)])] = 1.
		# summed source for normalization
		m_x = m.sum(axis=-1, keepdims=True)
		
		# compute p_dot
		p_dot = a*m - p * (a*m_x + b*g2_tilde(p)- c*g1_tilde(p))

		# set p_dot to 0 when P(y|x) is near 0 or 1.
		# This avoids numerical issues
		diffs = p[:,:,1]-p[:,:,0]
		signed_diffs = np.sign(p_dot[:,:,1]) * diffs
		p_dot = p_dot * (signed_diffs < (1-1e-6))[:,:,None]


		return p_dot.reshape(2*N**2)

	res = solve_ivp(fun, (0,T), p0, t_eval=t_eval,
		first_step=.1, rtol=1e-8, atol=1e-8)

	return res['t'], np.moveaxis(res['y'],0,1)[:,:2*N**2].reshape((T,N,N,2)), res['message']