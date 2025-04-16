import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.colors as mcolors
import matplotlib as mpl
from pathlib import Path
from dynamics_functions import *
from tqdm import tqdm

mpl.rcParams['axes.linewidth'] = 2


def single_plot(a, b, c, sigma, beta, N=100, T=1000, p0=None):
	"""
	Run a single associative learning experiment and plot task accuracy over time.

	Params
	------
	a : float
		Parameter controlling influence of new data samples.
	b : float
		Parameter controlling allowed distribution volatility.
	c : float
		Parameter controlling expected mutual information.
	sigma : float
		Controls volatility of true distribution. Higher sigma
		means less volatile.
	beta : float
		Controls uncertainty in true distribution. Higher beta implies
		less uncertainty (more biased towards 0 or 1)
	N : int
		Size of the grid on each side
	T : float
		Time to run integration
	p0 : np array (N x N x 2) or None
		Initial distribution, if None then initialize with the uniform
		distribution.
	"""
	T = int(T)
	# Generate true distribution
	truth = gen_dist(sigma, beta, N=N)
	# Generate 2*T samples from the ground truth
	samples = gen_samples(2*T,truth)

	# Get results
	t,ps,msg = get_dist(samples, a, b, c, p0=p0, N=N, T=T)

	# probability of correct classification over time
	accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))

	fig = plt.figure(constrained_layout=True, figsize=(3.5,2.1))
	axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})
	ax_res = axs['Left']
	ax_res.plot(t, accuracy, 'k', linewidth=6)
	ax_res.set_ylim(top=1.0)
	ax_res.set_xlabel('Time')
	ax_res.set_ylabel('Accuracy')

	axs['TopRight'].imshow(truth[:,:,1], cmap='coolwarm',vmin=0, vmax=1, origin='lower')
	axs['TopRight'].annotate(r'$P^{*}$', (1.1,0.0), xycoords='axes fraction', fontsize=12)
	axs['TopRight'].set_xticks([])
	axs['TopRight'].set_yticks([])

	axs['BottomRight'].imshow(ps[-1,:,:,1], cmap='coolwarm',vmin=0, vmax=1, origin='lower')
	axs['BottomRight'].annotate(r'$\hat P$', (1.1,0.0), xycoords='axes fraction', fontsize=12)
	axs['BottomRight'].set_xticks([])
	axs['BottomRight'].set_yticks([])

	fig.savefig('fig1b.png', format='png')


def collect_data_param_variation(a_lst, b_lst, c_lst, sigma_lst, beta_lst, N=100, T=3000):
	"""
	Compute how varying equation of motion, and task parameters changes transition time,
	abruptness and final accuracy. Compare to the analytically derived scaling laws.

	The raw data is stored in .npy files to be used later on.


	Params
	------
	a_lst : np array
		Array of a values to test. 0th element is the anchor value when varying b,c.
	b_lst : np array
		Array of b values to test. 0th element is the anchor value when varying a,c.
	c_lst : np array
		Array of c values to test. 0th element is the anchor value when varying a,b.
	sigma_lst : np array
		Array of sigma values to test, controlling task smoothness. 0th element is anchor.
	beta_lst : np array
		Array of beta values to test, controlling task uncertainty. 0th element is anchor.
	N : int
		number of bins in each x axis
	T : float
		amount of time to run integration
	"""
	data_folder = Path("param_vary_data")
	data_folder.mkdir(parents=True, exist_ok=True)

	T = int(T)
	# Default values
	a = a_lst[0]
	b = b_lst[0]
	c = c_lst[0]
	sigma = sigma_lst[0]
	beta = beta_lst[0]
	
	# Compute a variations
	ps_vary_a = []
	acc_vary_a = []

	# iterate through a_lst
	for a_iter in tqdm(a_lst):
		# Generate true distribution
		truth = gen_dist(sigma, beta, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a_iter, b, c, N=N, T=T)

		# compute probability of correct classification over time
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))

		ps_vary_a.append(ps)
		acc_vary_a.append(accuracy)
	# save results
	np.save("param_vary_data/ps_vary_a", ps_vary_a)
	np.save("param_vary_data/acc_vary_a", acc_vary_a)
	np.save("param_vary_data/a_lst", a_lst)

	# Compute b variations
	ps_vary_b = []
	acc_vary_b = []

	# iterate through b_lst
	for b_iter in tqdm(b_lst):
		# Generate true distribution
		truth = gen_dist(sigma, beta, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a, b_iter, c, N=N, T=T)

		# compute probability of correct classification over time
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))

		ps_vary_b.append(ps)
		acc_vary_b.append(accuracy)
	# save results
	np.save("param_vary_data/ps_vary_b", ps_vary_b)
	np.save("param_vary_data/acc_vary_b", acc_vary_b)
	np.save("param_vary_data/b_lst", b_lst)
	
	# Compute c variations
	ps_vary_c = []
	acc_vary_c = []

	# iterate through c_lst
	for c_iter in tqdm(c_lst):
		# Generate true distribution
		truth = gen_dist(sigma, beta, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a, b, c_iter, N=N, T=T)

		# compute probability of correct classification over time
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))

		ps_vary_c.append(ps)
		acc_vary_c.append(accuracy)
	# save results
	np.save("param_vary_data/c_lst", c_lst)
	np.save("param_vary_data/acc_vary_c", acc_vary_c)
	np.save("param_vary_data/ps_vary_c", ps_vary_c)


	# Compute sigma variations
	ps_vary_sigma = []
	acc_vary_sigma = []
	# iterate through sigma_lst
	for sigma_iter in tqdm(sigma_lst):
		# Generate true distribution
		truth = gen_dist(sigma_iter, beta, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a, b, c, N=N, T=T)

		# compute probability of correct classification over time
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))

		ps_vary_sigma.append(ps)
		acc_vary_sigma.append(accuracy)
	# save results
	np.save("param_vary_data/sigma_lst", sigma_lst)
	np.save("param_vary_data/acc_vary_sigma", acc_vary_sigma)
	np.save("param_vary_data/ps_vary_sigma", ps_vary_sigma)
	

	# Compute beta variations
	ps_vary_beta = []
	acc_vary_beta = []
	# iterate through beta list
	for beta_iter in tqdm(beta_lst):
		# Generate true distribution
		truth = gen_dist(sigma, beta_iter, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a, b, c, N=N, T=T)

		# compute probability of correct classification over time
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))

		ps_vary_beta.append(ps)
		acc_vary_beta.append(accuracy)
	# save results
	np.save("param_vary_data/beta_lst", beta_lst)
	np.save("param_vary_data/acc_vary_beta", acc_vary_beta)
	np.save("param_vary_data/ps_vary_beta", ps_vary_beta)

def plot_param_variation():
	"""
	Given the learning curve data under parameter variations, plot how
	latency, abruptness and accuracy vary with respect to each parameter. 

	Latency is defined as the time where the transition is (e-1)/(e+1) fraction
	complete. 
	Abruptness is the inverse time it takes for the transition to go from .25
	to .75 complete.

	The data paths are automatically chosen as the output paths in the 
	collect_data_param_variation function. 
	"""

	# define functions to compute latency and abruptness
	def compute_latency(ps):
		"""
		Given the learned conditional distribution over time,
		compute the onset latency of the transition.

		Onset latency is the time it takes for the transition to
		be (e-1)/(e+1) fraction complete.

		Definitions
		-----------
		M : Number of trajectories to compute latency for
		T : Number of time steps per trajectory
		N : The cue, x belongs to an N x N grid

		Params
		------
		ps : np array (M x T x N x N x 2)
			Array of M learned P(y|x) trajectories over time.

		Returns
		-------
		latencies : np array (M)
			The onset latency of the transition for each trajectory.

		"""

		# Compute fractional difference between each dist and the final
		pf = ps[:,None,-1] # final dists for each trajectory
		pi = ps[:,None,0] # initial dists for each trajectory
		fraction_diffs = np.abs(pf - ps).sum(axis=(2,3,4))/np.abs(pf-pi).sum(axis=(2,3,4))

		# Compute latencies as the first time when fractional diff > cutoff
		latencies = np.argmax(fraction_diffs > (np.e-1)/(np.e+1), axis=1)

		return latencies

	def compute_abruptness(ps):
		"""
		Given the learned conditional distribution over time,
		compute the abruptness of the transition.

		Abruptness is defined as the inverse of the time it takes
		for the transition to go from .25 to .75 fraction complete.

		Definitions
		-----------
		M : Number of trajectories to compute latency for
		T : Number of time steps per trajectory
		N : The cue, x belongs to an N x N grid

		Params
		------
		ps : np array (M x T x N x N x 2)
			Array of M learned P(y|x) trajectories over time.

		Returns
		-------
		abruptness : np array (M)
			The abruptness of the transition for each trajectory.
		"""

		# Compute fractional difference between each dist and the final
		pf = ps[:,None,-1] # final dists for each trajectory
		pi = ps[:,None,0] # initial dists for each trajectory
		fraction_diffs = np.abs(pf - ps).sum(axis=(2,3,4))/np.abs(pf-pi).sum(axis=(2,3,4))

		# Compute .25 points
		first_pts = np.argmax(fraction_diffs > .25, axis=1)
		# Compute .75 points
		second_pts = np.argmax(fraction_diffs > .75, axis=1)

		abruptness = 1/(second_pts - first_pts)

		return abruptness

	# get data for variations
	a_lst = np.load("param_vary_data/a_lst.npy")
	a_ps = np.load("param_vary_data/ps_vary_a.npy")
	a_acc = np.load("param_vary_data/acc_vary_a.npy")

	b_lst = np.load("param_vary_data/b_lst.npy")
	b_ps = np.load("param_vary_data/ps_vary_b.npy")
	b_acc = np.load("param_vary_data/acc_vary_b.npy")

	c_lst = np.load("param_vary_data/c_lst.npy")
	c_ps = np.load("param_vary_data/ps_vary_c.npy")
	c_acc = np.load("param_vary_data/acc_vary_c.npy")

	sigma_lst = np.load("param_vary_data/sigma_lst.npy")
	sigma_ps = np.load("param_vary_data/ps_vary_sigma.npy")
	sigma_acc = np.load("param_vary_data/acc_vary_sigma.npy")

	beta_lst = np.load("param_vary_data/beta_lst.npy")
	beta_ps = np.load("param_vary_data/ps_vary_beta.npy")
	beta_acc = np.load("param_vary_data/acc_vary_beta.npy")

	# First plot the a,b,c variation figure.

	fig,axes = plt.subplots(1,3, figsize=(6.3,2.1))
	scale_rates=1000
	rates_ylim = (0.002, 0.018)
	times_ylim = (400,1800)
	acc_ylim = (0.75,0.9)

	# define fitting functions
	def f0(a,k0):
		"""
		Compute transition time as a function of a and fitted k0.
		c_lst[0] is the default cvalue when calling the 
		plot_param_variation function.
		"""
		return -1/c_lst[0] * np.log(k0*a)

	def f1(c, k1):
		"""
		Compute transition rate as a function of c and fitted k1.
		"""
		return k1*c

	def f2(c,k0):
		"""
		Compute transition time as a funciton of c.
		"""
		return -np.log(k0*a_lst[0])/c


	# Compute latencies for a
	latencies_a = compute_latency(a_ps)
	# fit to latencies
	k0 = curve_fit(f0, a_lst, latencies, p0=.01)[0][0]
	print(r'k0 = {:.5e}'.format(k0), flush=True)
	fitted_curve_f0 = f0(a_lst, k0)

	# compute abruptness for a
	abruptness_a = compute_abruptness(a_ps)

	# Plot latency, abruptness, final accuracy vs a
	axes[0].set_xlabel(r'Data Influence ($a$)')
	axes[0].set_xscale('log')
	axes[0].set_ylabel('Latency', color='red')
	axes[0].scatter(a_lst, latencies_a, c='red', marker='o', alpha=.5)
	label = r'$-\frac{1}{c}\log(k_0a)$'
	axes[0].plot(a_lst[1:], fitted_curve_f0[1:], 'r-', label=label)
	axes[0].set_ylim(*times_ylim)
	axes[0].annotate(label, (.3*10**(-4), 1550), color='red', fontsize=13)

	twin_ax0 = axes[0].twinx()
	twin_ax0.scatter(a_lst, abruptness_a, c='blue', marker='o', alpha=.5)
	twin_ax0.set_yticks([])
	twin_ax0.set_ylim(*rates_ylim)

	acc_ax0 = axes[0].twinx()
	acc_ax0.set_yticks([])
	acc_ax0.scatter(a_lst, a_acc[:,-1], c='black', marker='o', alpha=.5)
	acc_ax0.set_ylim(*acc_ylim)

	
	# Compute latencies for b
	latencies_b = compute_latency(b_ps)

	# compute abruptness for b
	abruptness_b = compute_abruptness(b_ps)

	# Plot transition rates and times vs b
	axes[1].set_xlabel(r'Diffusion ($b$)')
	axes[1].set_xscale('log')
	axes[1].set_yticks([])
	axes[1].scatter(b_lst, latencies_b, c='red', marker='o', alpha=.5)
	axes[1].set_ylim(*times_ylim)

	twin_ax1 = axes[1].twinx()
	twin_ax1.set_yticks([])
	twin_ax1.scatter(b_lst, abruptness_b, c='blue', marker='o', alpha=.5)
	twin_ax1.set_ylim(*rates_ylim)

	acc_ax1 = axes[1].twinx()
	acc_ax1.set_yticks([])
	acc_ax1.scatter(b_lst, b_acc[:,-1], c='black', marker='o', alpha=.5)
	acc_ax1.set_ylim(*acc_ylim)


	# Compute latencies for c
	latencies_c = compute_latency(c_ps)
	# Compute fitted curve using the fitted parameter from a
	fitted_curve_f2 = f2(c_lst, k0)

	# compute abruptness for c
	abruptness_c = compute_abruptness(c_ps)

	# fit to transition rates
	k1 = curve_fit(f1, c_lst, abruptness_c, p0=.01)[0][0]
	fitted_curve_f1 = f1(c_lst, k1)
	print('k1 = {:.5}'.format(k1), flush=True)

	# Plot abruptness and latencies vs c
	axes[2].set_xlabel(r'Predictivity ($c$)')
	axes[2].set_yticks([])
	axes[2].scatter(c_lst, latencies_c, c='red', marker='o', alpha=.5)
	label = r'$-\frac{1}{c}\log (k_0 a)$'
	axes[2].plot(c_lst[1:], fitted_curve_f2[1:], 'r-', label=label)
	axes[2].set_ylim(*times_ylim)
	axes[2].annotate(label, (0.0162, 800), color='red', fontsize=12)

	twin_ax2 = axes[2].twinx()
	twin_ax2.set_ylabel(r'Abruptness ($\times 10^{-3}$)', color='blue')
	twin_ax2.scatter(c_lst, abruptness_c, c='blue', marker='o', alpha=.5)
	label = r'$ k_1c$'
	twin_ax2.plot(c_lst[1:], fitted_curve_f1[1:], 'b-', label=label)
	twin_ax2.set_ylim(*rates_ylim)
	# rescale abruptness so that it looks nicer
	ticks_y = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale_rates))
	twin_ax2.yaxis.set_major_formatter(ticks_y)
	axes[2].annotate(label, (0.0168,1075), color='blue', fontsize=12)

	acc_ax2 = axes[2].twinx()
	acc_ax2.spines['right'].set_position(('axes', 1.35))
	acc_ax2.set_ylabel('Final Accuracy', color='black')
	acc_ax2.scatter(c_lst, c_acc[:,-1], c='black', marker='o', alpha=.5)
	acc_ax2.set_ylim(*acc_ylim)

	# save figure
	fig.savefig("fig2a.png", format='png')

	# create figure for task variation
	fig_task,axes_task = plt.subplots(1,2, figsize=(4.2,2.1))

	# Compute latencies for sigma
	latencies_sigma = compute_latency(sigma_ps)
	# compute abruptness for sigma
	abruptness_a = compute_abruptness(sigma_ps)

	# Plot latency, abruptness, final accuracy vs sigma
	axes_task[0].set_xlabel(r'Task Smoothness ($\sigma$)')
	axes_task[0].set_ylabel('Latency', color='red')
	axes_task[0].scatter(sigma_lst, latencies_sigma, c='red', marker='o', alpha=.5)
	axes_task[0].set_ylim(*times_ylim)

	twin_ax0 = axes_task[0].twinx()
	twin_ax0.scatter(sigma_lst, abruptness_sigma, c='blue', marker='o', alpha=.5)
	twin_ax0.set_yticks([])
	twin_ax0.set_ylim(*rates_ylim)

	acc_ax0 = axes_task[0].twinx()
	acc_ax0.set_yticks([])
	acc_ax0.scatter(sigma_lst, sigma_acc[:,-1], c='black', marker='o', alpha=.5)
	acc_ax0.set_ylim(*acc_ylim)

	# Plot latency, abruptness, final accuracy vs beta
	# Compute latencies for beta
	latencies_beta = compute_latency(beta_ps)

	# compute abruptness for beta
	abruptness_beta = compute_abruptness(beta_ps)

	# Plot abruptness, latency and accuracy vs beta
	axes_task[1].set_xlabel(r'Uncertainty ($\beta^{-1}$)')
	axes_task[1].set_yticks([])
	axes_task[1].scatter(1/beta_lst, latencies_beta, c='red', marker='o', alpha=.5)
	axes_task[1].set_ylim(*times_ylim)

	twin_ax1 = axes_task[1].twinx()
	twin_ax1.set_ylabel(r'Abruptness ($\times 10^{-3}$)', color='blue')
	twin_ax1.scatter(1/beta_lst, abruptness_beta, c='blue', marker='o', alpha=.5)
	twin_ax1.set_ylim(*rates_ylim)
	# rescale abruptness so that it looks nicer
	ticks_y = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale_rates))
	twin_ax1.yaxis.set_major_formatter(ticks_y)

	acc_ax1 = axes_task[1].twinx()
	acc_ax1.spines['right'].set_position(('axes', 1.35))
	acc_ax1.set_ylabel('Final Accuracy', color='black')
	acc_ax1.scatter(1/beta_lst, beta_acc[:,-1], c='black', marker='o', alpha=.5)
	acc_ax1.set_ylim(*acc_ylim)

	# save figure
	fig_task.savefig("fig2b.png", format='png')

def plot_prior():
	"""
	Plot the prior over P(y|x) for discrete x=0,1 where P(x)=0.5.
	In this case, there is no volatility function and the prior is just
	exp(MI) where MI is the mutual information between X and Y. 

	This plot is a heatmap which shows that it is impossible to go from
	one high likelihood region to another without crossing a low-likelihood
	region. Thus, making the set of allowed hypotheses effectibely discrete.
	"""

	# Define binary entropy function h(p) = -p*ln(p) - (1-p)*ln(1-p)
	def h(p):
	    """Calculates binary entropy, handling p=0 and p=1 cases."""
	    p = np.asarray(p) # Ensure input is array for vectorized ops
	    # Use np.where to handle 0*log(0) = 0
	    term1 = np.where(p == 0, 0.0, p * np.log(p))
	    term2 = np.where(p == 1, 0.0, (1 - p) * np.log(1 - p))
	    # Entropy is non-negative, ensure result is >= 0
	    result = -(term1 + term2)
	    return np.maximum(0.0, result) # Clamp potential small negative due to precision

	# Define the function V(p0, p1) = exp(I[X;Y])
	def V(p0, p1):
	    """Calculates exp(Mutual Information) for given P(y=1|x=0) and P(y=1|x=1)."""
	    p0 = np.asarray(p0)
	    p1 = np.asarray(p1)

	    # Calculate marginal entropy H[Y]
	    # P(y=1) = 0.5 * (p0 + p1)
	    H_Y = h(0.5 * (p0 + p1))

	    # Calculate conditional entropy H[Y|X] = 0.5*h(p0) + 0.5*h(p1)
	    H_YcondX = 0.5 * h(p0) + 0.5 * h(p1)

	    # Calculate mutual information I[X;Y] = H[Y] - H[Y|X]
	    # Ensure non-negative due to potential floating point inaccuracies near zero
	    IMI = np.maximum(0.0, H_Y - H_YcondX)

	    # Return exponentiated value
	    return np.exp(IMI)

	# --- Set up grid ---
	num_points = 200 # Resolution of the grid
	p0_vals = np.linspace(0, 1, num_points)
	p1_vals = np.linspace(0, 1, num_points)
	# Create 2D grid of coordinates
	# Note: meshgrid indexing default is 'xy', P0 varies along x (columns), P1 along y (rows)
	P0, P1 = np.meshgrid(p0_vals, p1_vals)

	# --- Calculate V over the grid ---
	V_grid = V(P0, P1)

	# --- Plotting ---
	fig, ax = plt.subplots(figsize=(3.5,2.92))

	# Define the custom white-to-red colormap
	cmap_white_red = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])

	# display heatmap as img
	im = ax.imshow(V_grid,
	               origin='lower',
	               extent=[0, 1, 0, 1],
	               cmap=cmap_white_red,
	               vmin=1, # Value corresponding to white
	               vmax=2, # Value corresponding to red
	               aspect='equal' # Ensure the plot area is square
	              )

	# set labels
	ax.set_xlabel(r"$P(y=1|x=0)$", fontsize=12)
	ax.set_ylabel(r"$P(y=1|x=1)$", fontsize=12)
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.tick_params(axis='both', which='major', labelsize=10)

	# Image border
	for spine in ax.spines.values():
	    spine.set_edgecolor('black')
	    spine.set_linewidth(2.0)

	# Colorbar
	cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
	cbar.set_label(r"$\exp(I[X;Y])$", fontsize=12, rotation=270, labelpad=15)
	# Set specific ticks on the colorbar
	cbar.set_ticks(np.linspace(1, 2, 6)) # e.g., 6 ticks from 1.0 to 2.0
	cbar.ax.tick_params(labelsize=10)


	fig.savefig("fig3a.png", format='png')

def plot_random_task(a, b, c, N=100, T=1000, p0=None):
	"""
	Run a single associative learning experiment with a random reward schedule.
	Plot change in distribution over time, with inset showing the final
	learned structure. 

	Params
	------
	a : float
		Parameter controlling influence of new data samples.
	b : float
		Parameter controlling allowed distribution volatility.
	c : float
		Parameter controlling expected mutual information.
	N : int
		Size of the grid on each side
	T : float
		Time to run integration
	p0 : np array (N x N x 2) or None
		Initial distribution, if None then initialize with the uniform
		distribution.
	"""

	T = int(T)
	# Generate fully random true distribution
	truth = 0.5 * np.ones((N,N,2))
	# Generate 2*T samples from the ground truth
	samples = gen_samples(2*T,truth)

	# Get results from learning with random samples
	t,ps,msg = get_dist(samples, a, b, c, p0=p0, N=N, T=T)

	# average magnitude of derivative of P_t(y|x) over time
	dp_dt = np.abs(np.diff(ps, axis=0)).mean(axis=(1,2,3))

	fig, ax = plt.subplots(figsize=(3.5,2.1))
	
	# Plot change in P over time
	ax.plot(t[1:], dp_dt, 'k', linewidth=3)
	ax.set_xlabel('Time')
	ax.set_ylabel(r'Avg. Change ($|d\hat P_t/dt|$)')

	# Create inset with final dist
	inset_bounds = [0.65, 0.65, 0.3, 0.3] # inset location, size
	axins = ax.inset_axes(inset_bounds)

	# show final output
	axins.imshow(ps[-1,:,:,1], cmap='coolwarm', vmin=0, vmax=1, origin='lower')
	axins.set_xticks([])
	axins.set_yticks([])

	# Draw connecting lines for inset
	# connection coords for bottom right
	xyA = (1.0, 0.0)
	# connection coords for bottom right and left of inset
	xyB_corner1 = (0.0, 0.0)
	xyB_corner2 = (1.0, 0.0)

	# Create first line to bottom-left of inset
	con1 = ConnectionPatch(
	    xyA=xyA, coordsA=ax.transAxes,
	    xyB=xyB_corner1, coordsB=axins.transAxes,
	    arrowstyle="-", 
	    linestyle="--",
	    color="gray",
	    linewidth=1.0
	)

	# Create second line to bottom right of inset
	con2 = ConnectionPatch(
	    xyA=xyA, coordsA=ax.transAxes,
	    xyB=xyB_corner2, coordsB=axins.transAxes,
	    arrowstyle="-",
	    linestyle="--",
	    color="gray",
	    linewidth=1.0
	)

	fig.add_artist(con1)
	fig.add_artist(con2)

	ax.set_clip_on(False)
	axins.set_clip_on(False)

	fig.savefig('fig3b.png', format='png')

	# return average change in probs at end of curve
	delta_p = np.abs(ps[int(T*0.75)] - ps[-1]).mean()

	return delta_p

