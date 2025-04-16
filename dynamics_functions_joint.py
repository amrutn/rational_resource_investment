import numpy as np
from scipy.integrate import solve_ivp

def create_smoothed_rectangle_array(N, w, h, high_val, low_val, transition_steps=3):
    """
    Helper function for generating P(x)
    Creates an N x N array with a central w x h rectangle of high_val
    and a surrounding area of low_val, with a smoothed transition zone.

    Params
    ------
    N : int
    	Size of the N x N array
    w : int
    	Width of the central high-value rectangle.
    h : int
    	Height of the central high-value rectangle.
    high_val : float
    	Value inside the central rectangle.
    low_val : float
    	Background value far from the rectangle.
    transition_steps : int
    	Number of steps over which the value
        transitions from high_val to low_val.

    Returns
    -------
    rectangle : np array (N x N)
    	The N x N array with the smoothed rectangle.
    """
    if w > N or h > N:
        raise ValueError("Rectangle dimensions (w, h) cannot exceed array size N")
    if transition_steps <= 0:
        raise ValueError("transition_steps must be positive")

    # Initialize array with low_val
    arr = np.full((N, N), low_val, dtype=float)

    # Calculate coordinates of the central rectangle (inclusive edges)
    vert = [N//2 - h//2, N//2 + h//2]
    horiz = [N//2 - w//2, N//2 + w//2]

    # Fill the high value rectangle directly first
    arr[horiz[0]:horiz[1], vert[0]:vert[1]] = high_val

    # Iterate through all cells to calculate smoothed values outside the rectangle
    for r in range(N):
        for c in range(N):
            # Skip cells already inside the high_val rectangle
            if horiz[0] <= r < horiz[1] and vert[0] <= c < vert[1]:
                continue

            # Calculate Chebyshev distance to the rectangle boundary
            if r < horiz[0]:
                dr = horiz[0] - r
            elif r > horiz[1]:
                dr = r - horiz[1] + 1
            else:
                dr = 0

            if c < vert[0]:
                dc = vert[0] - c
            elif c > vert[1]:
                dc = c - vert[1] + 1
            else:
                dc = 0

            # Chebyshev distance is the max of the two
            dist = max(dr, dc)

            # Calculate the smoothed value based on distance
            if dist <= transition_steps:
                 # Linear interpolation from high_val down to low_val over transition_steps
                 value = low_val + (high_val-low_val) * (1-dist/transition_steps)
            else:
                # Beyond the transition zone, it's just low_val (already set)
                arr[r, c] = low_val

    return arr

# generate a ground truth joint distribution
def gen_dist_x(w, h, p_high=0.99, N=100):
	"""
	Generate P(x). This is done by first creating
	a rectangle of width w and height h, centered at (N/2,N/2) such that
	P(x) is uniform in that rectangle and the probability of the rectangle
	is p_x. The probability outside the rectangle is typically small compared
	to inside.

	Params
	------
	sigma : float
		Standard deviation of the Gaussian smoothing filter, 
		controls volatility. Larger sigma = smoother field for P(y|x).
	beta : float
		Gain of the sigmoid nonlinearity, controls uncertainty.
		Larger gain = probabilities closer to 0 or 1 for P(y|x).
	w : int
		Width of the rectangle for the high-likelihood region of P(x)
	h : int
		Height of the rectangle for the high-likelihood region of P(x)
	p_high : float
		The probability of the high-likelihood region of P(x)
	N : int
		Size of the grid on each side
	Returns
	-------
	joint_dist : np array (N x N x 1)
		Generated P(x)
	"""

	assert w < N-10
	assert h < N-10

	p_x = create_smoothed_rectangle_array(N, w, h, high_val=p_high/(w*h), low_val=(1-p_high)/(N**2-w*h), transition_steps=5)
	p_x /= p_x.sum() # enforcing exact normalization

	return p_x[:,:,None]

# Compute Mutual Information of joint dist
def compute_mi_joint(px,py_x):
	"""
	Compute mutual information of joint distribution p, defined by g_1[p]

	Params
	------
	px : np array (N x N x 1)
		P(x)
	py_x : np array(N x N x 2)
		P(y|x)

	Returns
	-------
	mi : float
		Mutual information
	"""
	eps = np.finfo(float).tiny
	p_y = (py_x*px).sum(axis=(0,1), keepdims=True)
	return (py_x * px * np.log(py_x+eps)).sum() - (p_y * np.log(p_y+eps)).sum()

# Compute volatility of joint dist
def compute_vol_joint(px,py_x):
	"""
	Compute the volatility of joint distribution p, defined by g_2[p]
	for discrete cues.

	Params
	------
	px : np array (N x N x 1)
		P(x)
	py_x : np array(N x N x 2)
		P(y|x)

	Returns
	-------
	vol : float
		Volatility
	"""
	eps = np.finfo(float).tiny

	p1y_x = np.roll(py_x,1,axis=0)
	p1y_x[0,:] = py_x[0,:] # Boundary condition

	p2y_x = np.roll(py_x,1,axis=1)
	p2y_x[:,0] = py_x[:,0] # Boundary condition

	p3y_x = np.roll(py_x,-1,axis=0)
	p3y_x[-1,:] = py_x[-1,:] # Boundary condition

	p4y_x = np.roll(py_x,-1,axis=1)
	p4y_x[:,-1] = py_x[:,-1] # Boundary condition

	return (px*py_x*(4*np.log(py_x+eps) - np.log(p1y_x+eps) - np.log(p2y_x+eps) - np.log(p3y_x+eps) - np.log(p4y_x+eps))).sum()



