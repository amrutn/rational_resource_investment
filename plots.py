import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl
from scipy.optimize import curve_fit
from pathlib import Path
from dynamics_functions import *
from tqdm import tqdm

mpl.rcParams['axes.linewidth'] = 2
plt.rcParams.update({
    "text.usetex": True
})


# Figure 1, heatmap of constraint functions for different P(x)
def px_vs_constraint(sigma, beta, p_high=0.99, N=100, nsamples=100, collect_data=True):
	"""
	For each possible perimeter and area of the high likelihood rectangle, as 
	controlled by its width and height, plot the value of the mutual information, volatility
	and their sum when sampling P(y|x) over fixed sigma, beta many times to avoid bias.

	Params
	------
	sigma : float
		Standard deviation of the Gaussian smoothing filter, 
		controls volatility. Larger sigma = smoother field for P(y|x).
	beta : float
		Gain of the sigmoid nonlinearity, controls uncertainty.
		Larger gain = probabilities closer to 0 or 1 for P(y|x).
	p_high : float
		The probability of the high-likelihood region of P(x)
	N : int
		Size of the grid on each side
	nsamples : int
		Number of times to sample each P(y|x)
	collect_data : bool
		If true, generate data for the plots
	"""
	if collect_data:
		data_folder = Path("px_data")
		data_folder.mkdir(parents=True, exist_ok=True)

		# create all the P(x) for each width and height
		# limit the range to avoid effects from tiny boxes
		# and from forcing p(Y)=.5 after marginalization
		x_dists = []
		entropies = []
		dimensions = []
		ws = []
		hs = []
		for h in range(int(.25*N),int(.75*N)):
			for w in range(2,int(.75*N)):
				px = gen_dist_x(w, h, p_high=p_high, N=N)
				x_dists.append(px)
				entropies.append(compute_entropy(px))
				dimensions.append(compute_d_eff_x(px))
				ws.append(w)
				hs.append(h)

		# compute MI and vol for each x dist over nsamples samples of P(y|x)
		all_mis = []
		all_vols = []
		for i in tqdm(range(nsamples)):
			py_x = gen_dist_cond(sigma, beta, N=N)
			mis = []
			vols = []
			for px in x_dists:
				mis.append(compute_mi_joint(px,py_x))
				vols.append(compute_vol_joint(px,py_x))
			all_mis.append(mis)
			all_vols.append(vols)


		all_mis = np.array(all_mis)
		all_vols = np.array(all_vols)

		np.save('px_data/mis', all_mis)
		np.save('px_data/vols', all_vols)
		np.save('px_data/ws', ws)
		np.save('px_data/hs', hs)
		np.save('px_data/entropies', entropies)
		np.save('px_data/dimensions', dimensions)

	mis = np.load('px_data/mis.npy').mean(axis=0)
	vols = np.load('px_data/vols.npy').mean(axis=0)
	entropies = np.load('px_data/entropies.npy')
	dimensions = np.load('px_data/dimensions.npy')


	# make figures for mis, vols
	mean_mi_dev = np.abs(mis-mis.mean()).mean()
	mean_vol_dev = np.abs(vols-vols.mean()).mean()

	combined = np.exp(mis-mean_mi_dev/mean_vol_dev*vols)

	fig,ax = plt.subplots(1,1, figsize=(2.5,1.7), layout="constrained")
	sc = ax.scatter(dimensions, entropies, 
                 c=combined,           
                 cmap='Purples',        # Choose a colormap
                 marker='s',            # Use square markers
                 s=2        # Set marker size
                )
	cbar = plt.colorbar(sc)
	cbar.set_label(r'Prior')
	ax.set_ylabel(r'Cost: $H[x]$')
	ax.set_xlabel(r'Effective Dimensionality')

	fig.savefig('fig2b.pdf', format='pdf', dpi=500)

	return mean_mi_dev/mean_vol_dev

# Figure 2b
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
	truth = gen_dist_cond(sigma, beta, N=N)
	# Generate 2*T samples from the ground truth
	samples = gen_samples_cond(2*T,truth)

	# Get results
	t,ps,msg = get_dist(samples, a, b, c, p0=p0, N=N, T=T)

	# probability of correct classification over time
	accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))

	fig = plt.figure(constrained_layout=True, figsize=(2.9,1.7))
	axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})
	ax_res = axs['Left']
	ax_res.plot(t, accuracy, 'k')
	ax_res.set_ylim(top=1.0)
	ax_res.set_xlabel('Time')
	ax_res.set_ylabel('Accuracy')

	axs['TopRight'].imshow(truth[:,:,1], cmap='coolwarm',vmin=0, vmax=1, origin='lower')
	axs['TopRight'].annotate("$P^{*}(1|x)$", (-0.05,-0.3), xycoords='axes fraction')
	axs['TopRight'].set_xticks([])
	axs['TopRight'].set_yticks([])

	axs['BottomRight'].imshow(ps[-1,:,:,1], cmap='coolwarm',vmin=0, vmax=1, origin='lower')
	axs['BottomRight'].annotate(r"$\hat P_{T}(1|x)$", (-0.05,-0.3), xycoords='axes fraction')
	axs['BottomRight'].set_xticks([])
	axs['BottomRight'].set_yticks([])

	fig.savefig('fig3b.pdf', format='pdf', dpi=500)


def collect_data_param_variation(a_lst, b_lst, c_lst, sigma_lst, beta_lst, num=30, N=100, T=3000):
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
	num : int
		Number of random simulations to do
	N : int
		number of bins in each x axis
	T : float
		amount of time to run integration
	"""
	data_folder = Path("param_vary_data")
	data_folder.mkdir(parents=True, exist_ok=True)

	# define functions to compute latency and abruptness
	def compute_latency(ps):
		"""
		Given the learned conditional distribution over time,
		compute the onset latency of the transition.

		Onset latency is the time it takes for the transition to
		be (e-1)/(e+1) fraction complete.

		Definitions
		-----------
		T : Number of time steps per trajectory
		N : The cue, x belongs to an N x N grid

		Params
		------
		ps : np array (T x N x N x 2)
			Learned P(y|x) over time

		Returns
		-------
		latency : int
			Onset latency of the trajectory.

		"""

		# Compute fractional difference between each dist and the final
		pf = ps[-1,None] # final dists for each trajectory
		pi = ps[0,None] # initial dists for each trajectory
		fraction_diffs = 1-np.abs(pf - ps).sum(axis=(1,2,3))/np.abs(pf-pi).sum(axis=(1,2,3))

		# Compute latencies as the first time when fractional diff > cutoff
		latency = np.argmax(fraction_diffs > (np.e-1)/(np.e+1))

		return latency

	def compute_abruptness(ps):
		"""
		Given the learned conditional distribution over time,
		compute the abruptness of the transition.

		Abruptness is defined as the inverse of the time it takes
		for the transition to go from .25 to .75 fraction complete.

		Definitions
		-----------
		T : Number of time steps per trajectory
		N : The cue, x belongs to an N x N grid

		Params
		------
		ps : np array (T x N x N x 2)
			Learned P(y|x) over time

		Returns
		-------
		abruptness : float
			The abruptness of the transition for each trajectory.
		"""

		# Compute fractional difference between each dist and the final
		pf = ps[-1,None] # final dists for each trajectory
		pi = ps[0,None] # initial dists for each trajectory
		fraction_diffs = 1-np.abs(pf - ps).sum(axis=(1,2,3))/np.abs(pf-pi).sum(axis=(1,2,3))

		# Compute .25 points
		first_pt = np.argmax(fraction_diffs > .25)
		# Compute .75 points
		second_pt = np.argmax(fraction_diffs > .75)

		abruptness = 1/(second_pt - first_pt)

		return abruptness

	T = int(T)
	# Default values
	a = a_lst[0]
	b = b_lst[0]
	c = c_lst[0]
	sigma = sigma_lst[0]
	beta = beta_lst[0]
	
	# Compute a variations
	latency_vary_a = []
	abruptness_vary_a = []
	acc_vary_a = []

	# iterate through a_lst
	print("Collecting a Parameter Variations")

	for a_iter in tqdm(a_lst):
		# Generate true distribution
		truth = gen_dist_cond(sigma, beta, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples_cond(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a_iter, b, c, N=N, T=T)

		# compute accuracy, latency, abruptness
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))
		latency = compute_latency(ps)
		abruptness = compute_abruptness(ps)

		latency_vary_a.append(latency)
		abruptness_vary_a.append(abruptness)
		acc_vary_a.append(accuracy)

	# save results
	np.save("param_vary_data/latency_vary_a", latency_vary_a)
	np.save("param_vary_data/abruptness_vary_a", abruptness_vary_a)
	np.save("param_vary_data/acc_vary_a", acc_vary_a)
	np.save("param_vary_data/a_lst", a_lst)

	# Compute b variations
	print("Collecting b Parameter Variations")

	latency_vary_b = []
	abruptness_vary_b = []
	acc_vary_b = []

	# iterate through b_lst
	for b_iter in tqdm(b_lst):
		# Generate true distribution
		truth = gen_dist_cond(sigma, beta, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples_cond(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a, b_iter, c, N=N, T=T)

		# compute accuracy, latency, abruptness
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))
		latency = compute_latency(ps)
		abruptness = compute_abruptness(ps)

		latency_vary_b.append(latency)
		abruptness_vary_b.append(abruptness)
		acc_vary_b.append(accuracy)

	# save results
	np.save("param_vary_data/latency_vary_b", latency_vary_b)
	np.save("param_vary_data/abruptness_vary_b", abruptness_vary_b)
	np.save("param_vary_data/acc_vary_b", acc_vary_b)
	np.save("param_vary_data/b_lst", b_lst)
	
	# Compute c variations
	print("Collecting c Parameter Variations")

	latency_vary_c = []
	abruptness_vary_c = []
	acc_vary_c = []

	# iterate through c_lst
	for c_iter in tqdm(c_lst):
		# Generate true distribution
		truth = gen_dist_cond(sigma, beta, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples_cond(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a, b, c_iter, N=N, T=T)

		# compute accuracy, latency, abruptness
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))
		latency = compute_latency(ps)
		abruptness = compute_abruptness(ps)

		latency_vary_c.append(latency)
		abruptness_vary_c.append(abruptness)
		acc_vary_c.append(accuracy)

	# save results
	np.save("param_vary_data/latency_vary_c", latency_vary_c)
	np.save("param_vary_data/abruptness_vary_c", abruptness_vary_c)
	np.save("param_vary_data/acc_vary_c", acc_vary_c)
	np.save("param_vary_data/c_lst", c_lst)
	

	# Compute sigma variations
	print("Collecting Sigma Parameter Variations")

	latency_vary_sigma = []
	abruptness_vary_sigma = []
	acc_vary_sigma = []
	g_1_lst = []

	# iterate through sigma_lst
	for sigma_iter in tqdm(sigma_lst):
		# Generate true distribution
		truth = gen_dist_cond(sigma_iter, beta, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples_cond(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a, b, c, N=N, T=T)

		# compute accuracy, latency, abruptness
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))
		latency = compute_latency(ps)
		abruptness = compute_abruptness(ps)

		latency_vary_sigma.append(latency)
		abruptness_vary_sigma.append(abruptness)
		acc_vary_sigma.append(accuracy)
		g_1_lst.append(-compute_vol_joint(1/N**2 * np.ones((N,N,1)), truth))

	# save results
	np.save("param_vary_data/latency_vary_sigma", latency_vary_sigma)
	np.save("param_vary_data/abruptness_vary_sigma", abruptness_vary_sigma)
	np.save("param_vary_data/acc_vary_sigma", acc_vary_sigma)
	np.save("param_vary_data/sigma_lst", sigma_lst)
	np.save("param_vary_data/g_1_lst", g_1_lst)
	

	# Compute beta variations
	print("Collecting Beta Parameter Variations")

	latency_vary_beta = []
	abruptness_vary_beta = []
	acc_vary_beta = []
	g_2_lst = []

	# iterate through beta_lst
	for beta_iter in tqdm(beta_lst):
		# Generate true distribution
		truth = gen_dist_cond(sigma, beta_iter, N=N)
		# Generate 2*T samples from the ground truth
		samples = gen_samples_cond(2*T,truth)

		# Get results
		t,ps,msg = get_dist(samples, a, b, c, N=N, T=T)

		# compute accuracy, latency, abruptness
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))
		latency = compute_latency(ps)
		abruptness = compute_abruptness(ps)

		latency_vary_beta.append(latency)
		abruptness_vary_beta.append(abruptness)
		acc_vary_beta.append(accuracy)
		g_2_lst.append(compute_mi_joint(1/N**2 * np.ones((N,N,1)), truth))

	# save results
	np.save("param_vary_data/latency_vary_beta", latency_vary_beta)
	np.save("param_vary_data/abruptness_vary_beta", abruptness_vary_beta)
	np.save("param_vary_data/acc_vary_beta", acc_vary_beta)
	np.save("param_vary_data/beta_lst", beta_lst)
	np.save("param_vary_data/g_2_lst", g_2_lst)
	
	# Collect data for random parameter variations
	print("Collecting Random Parameter Variations")
	# Compute mean landa_eta, lambda_1, lambda_2
	landa_eta_mean = N**2/a
	lambda_1_mean = c * landa_eta_mean
	lambda_2_mean = b/2 * landa_eta_mean
	
	# Sample 25 random parameters from an exponential distribution
	landa_etas = get_exp_params(landa_eta_mean, num)
	lambda_1s = get_exp_params(lambda_1_mean, num)
	lambda_2s = get_exp_params(lambda_2_mean, num)

	# True distribution and samples
	truth = gen_dist_cond(sigma, beta, N=N)
	samples = gen_samples_cond(2*T,truth)

	# Run simulations with the parameter combinations
	latency_random = []
	abruptness_random = []
	accuracies_random = []
	for i in tqdm(range(num)):
		tmp_a = N**2/landa_etas[i]
		tmp_b = 2*lambda_2s[i]/landa_etas[i]
		tmp_c = lambda_1s[i]/landa_etas[i]

		# Get results
		t,ps,msg = get_dist(samples, tmp_a, tmp_b, tmp_c, N=N, T=T)

		# compute accuracy, latency, abruptness
		accuracy = (truth*ps).sum(axis=-1).mean(axis=(1,2))
		if accuracy[-1] > 0.75: # only include data which transitioned
			latency_random.append(compute_latency(ps))
			abruptness_random.append(compute_abruptness(ps))
		if accuracy.size == int(T):
			accuracies_random.append(accuracy)

	# save results
	np.save("param_vary_data/latency_random", latency_random)
	np.save("param_vary_data/abruptness_random", abruptness_random)
	np.save("param_vary_data/accuracies_random", accuracies_random)
	
def plot_random_curves():
	"""
	Plot learning curves with exponentially distributed parameters
	and mean curve, to compare with Gallistel's figure.
	"""
	accuracies = np.load("param_vary_data/accuracies_random.npy", allow_pickle=True)
	mean_curve = accuracies.mean(axis=0)

	fig,ax=plt.subplots(figsize=(2.1,1.7), layout="constrained")
	for i in range(10):
		T = accuracies[i].size
		if i ==0:
			ax.plot(range(T),accuracies[i], 'k', alpha=.4, label='Individual')
		else:
			ax.plot(range(T),accuracies[i], 'k', alpha=.4)

	ax.plot(range(T), mean_curve, 'k', linewidth=3, label='Averaged')
	ax.set_ylim(top=1)
	ax.set_ylabel("Accuracy")
	ax.set_xlabel("Time")
	# save figure
	fig.savefig("fig5.pdf", format='pdf', dpi=500)
	return

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
	# First plot the a,b,c variation figure.

	fig,axes = plt.subplots(1,3, figsize=(5.4,1.7), layout="constrained")
	scale_rates=1000
	rates_ylim = (0.002, 0.015)
	times_ylim = (.4,1.75)
	acc_ylim = (0.5,1.0)

	# define fitting functions
	def f0(a_iter,k0):
		"""
		Compute transition time as a function of changing a and fitted k0.
		Fixed default c.
		"""
		return -1/c_lst[0] * np.log(k0*a_iter)

	def f1(c_iter, k1):
		"""
		Compute transition rate as a function of changing c and fitted k1.
		"""
		return k1*c_iter

	def f2(c_iter,k0):
		"""
		Compute transition time as a funciton of changing c, fixed a.
		"""
		return -np.log(k0*a_lst[0])/c_iter

	def f3(rho,k0,k1):
		"""
		tau vs rho
		"""
		return -np.log(k0*a_lst[0]) * k1 * rho**(-1)

	# Load all the data
	a_lst = np.load("param_vary_data/a_lst.npy")
	b_lst = np.load("param_vary_data/b_lst.npy")
	c_lst = np.load("param_vary_data/c_lst.npy")
	sigma_lst = np.load("param_vary_data/sigma_lst.npy")
	beta_lst = np.load("param_vary_data/beta_lst.npy")
	g_1_lst = np.load("param_vary_data/g_1_lst.npy")
	g_2_lst = np.load("param_vary_data/g_2_lst.npy")

	abruptness_a = np.load("param_vary_data/abruptness_vary_a.npy")
	latencies_a = np.load("param_vary_data/latency_vary_a.npy")
	a_acc = np.load("param_vary_data/acc_vary_a.npy")

	abruptness_b = np.load("param_vary_data/abruptness_vary_b.npy")
	latencies_b = np.load("param_vary_data/latency_vary_b.npy")
	b_acc = np.load("param_vary_data/acc_vary_b.npy")

	abruptness_c = np.load("param_vary_data/abruptness_vary_c.npy")
	latencies_c = np.load("param_vary_data/latency_vary_c.npy")
	c_acc = np.load("param_vary_data/acc_vary_c.npy")

	abruptness_sigma = np.load("param_vary_data/abruptness_vary_sigma.npy")
	latencies_sigma = np.load("param_vary_data/latency_vary_sigma.npy")
	sigma_acc = np.load("param_vary_data/acc_vary_sigma.npy")

	abruptness_beta = np.load("param_vary_data/abruptness_vary_beta.npy")
	latencies_beta = np.load("param_vary_data/latency_vary_beta.npy")
	beta_acc = np.load("param_vary_data/acc_vary_beta.npy")

	abruptness_random = np.load("param_vary_data/abruptness_random.npy")
	latencies_random = np.load("param_vary_data/latency_random.npy")

	# fit to latencies
	k0 = curve_fit(f0, a_lst, latencies_a, p0=.01)[0][0]
	fitted_curve_f0 = f0(a_lst, k0)
	# computing r^2
	SS_res = ((latencies_a - fitted_curve_f0)**2).sum()
	SS_tot = ((latencies_a - latencies_a.mean())**2).sum()
	r_square_f0 = 1-SS_res/SS_tot

	# Plot latency, abruptness, final accuracy vs a
	axes[0].set_xlabel(r"Data ($a$)")
	axes[0].set_xscale('log')
	axes[0].set_ylabel(r'Latency ($\times 10^3$)', color='red')
	axes[0].tick_params(axis='y', colors='red')
	axes[0].scatter(a_lst, latencies_a/10**3, c='red', marker='o', alpha=.5, s=15)
	label = r'$-\frac{1}{c}\log(k_0a)$'
	axes[0].plot(a_lst[1:], fitted_curve_f0[1:]/10**3, 'r-', label=label)
	axes[0].set_ylim(*times_ylim)
	axes[0].annotate(label, (.22,.1), xycoords='axes fraction', color='red')

	twin_ax0 = axes[0].twinx()
	twin_ax0.scatter(a_lst, abruptness_a, c='blue', marker='o', alpha=.5, s=15)
	twin_ax0.set_yticks([])
	twin_ax0.set_ylim(*rates_ylim)

	acc_ax0 = axes[0].twinx()
	acc_ax0.set_yticks([])
	acc_ax0.scatter(a_lst, a_acc[:,-1], c='black', marker='o', alpha=.5, s=15)
	acc_ax0.set_ylim(*acc_ylim)


	# Plot transition rates and times vs b
	axes[1].set_xlabel(r'Smoothness ($b$)')
	axes[1].set_xscale('log')
	axes[1].set_yticks([])
	axes[1].scatter(b_lst, latencies_b/10**3, c='red', marker='o', alpha=.5, s=15)
	axes[1].set_ylim(*times_ylim)
	argmin_lat = b_lst[np.argmin(latencies_b)]
	axes[1].vlines(argmin_lat, ymin=0, ymax=(latencies_b/10**3).min(), alpha=.4, colors='r', linestyles='--', linewidth=.5)

	twin_ax1 = axes[1].twinx()
	twin_ax1.set_yticks([])
	twin_ax1.scatter(b_lst, abruptness_b, c='blue', marker='o', alpha=.5, s=15)
	twin_ax1.set_ylim(*rates_ylim)
	argmax_abrupt = b_lst[np.argmax(abruptness_b)]
	twin_ax1.vlines(argmax_abrupt, ymin=0, ymax=abruptness_b.max(), alpha=.4, colors='b', linestyles='--', linewidth=.5)

	acc_ax1 = axes[1].twinx()
	acc_ax1.set_yticks([])
	acc_ax1.scatter(b_lst, b_acc[:,-1], c='black', marker='o', alpha=.5, s=15)
	acc_ax1.set_ylim(*acc_ylim)
	argmax_acc = b_lst[np.argmax(b_acc[:,-1])]
	twin_ax1.vlines(argmax_acc, ymin=0, ymax=b_acc[:,-1].max(), alpha=.4, colors='k', linestyles='--', linewidth=.5)

	# Compute fitted curve using the fitted parameter from a
	fitted_curve_f2 = f2(c_lst, k0)
	# computing r^2
	SS_res = ((latencies_c[3:] - fitted_curve_f2[3:])**2).sum()
	SS_tot = ((latencies_c[3:] - latencies_c[3:].mean())**2).sum()
	r_square_f2 = 1-SS_res/SS_tot

	# fit to abruptness
	k1 = curve_fit(f1, c_lst, abruptness_c, p0=.01)[0][0]
	fitted_curve_f1 = f1(c_lst, k1)
	# computing r^2
	SS_res = ((abruptness_c - fitted_curve_f1)**2).sum()
	SS_tot = ((abruptness_c- abruptness_c.mean())**2).sum()
	r_square_f1 = 1-SS_res/SS_tot

	# Plot abruptness and latencies vs c
	axes[2].set_xlabel(r'Mutual Information ($c$)')
	axes[2].set_yticks([])
	axes[2].scatter(c_lst, latencies_c/10**3, c='red', marker='o', alpha=.5, s=15)
	axes[2].plot(c_lst[1:], fitted_curve_f2[1:]/10**3, 'r-', label=label)
	axes[2].set_ylim(*times_ylim)
	label = r'$-\frac{1}{c}\log(k_0a)$'
	axes[2].annotate(label, (.15,.1), xycoords='axes fraction', color='red')

	twin_ax2 = axes[2].twinx()
	twin_ax2.set_ylabel(r'Abruptness ($\times 10^{-3}$)', color='blue')
	twin_ax2.scatter(c_lst, abruptness_c, c='blue', marker='o', alpha=.5, s=15)
	twin_ax2.tick_params(axis='y', colors='blue')
	label = r'$k_1c$'
	twin_ax2.plot(c_lst[1:], fitted_curve_f1[1:], 'b-', label=label)
	twin_ax2.set_ylim(*rates_ylim)
	# rescale abruptness so that it looks nicer
	ticks_y = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale_rates))
	twin_ax2.yaxis.set_major_formatter(ticks_y)
	axes[2].annotate(label, (0.78,0.5), xycoords='axes fraction', color='blue')

	acc_ax2 = axes[2].twinx()
	acc_ax2.spines['right'].set_position(('axes', 1.4))
	acc_ax2.set_ylabel('Final Accuracy', color='black')
	acc_ax2.scatter(c_lst, c_acc[:,-1], c='black', marker='o', alpha=.5, s=15)
	acc_ax2.set_ylim(*acc_ylim)


	# save figure
	fig.savefig("fig4a.pdf", format='pdf', dpi=500)

	# create figure for task variation
	fig_task,axes_task = plt.subplots(1,2, figsize=(3.7,1.7), layout="constrained")


	# Plot latency, abruptness, final accuracy vs sigma
	axes_task[0].set_xlabel(r'$\exp(g_1[P^*])$')
	axes_task[0].set_ylabel(r'Latency ($\times 10^3$)', color='red')
	axes_task[0].tick_params(axis='y', colors='red')
	axes_task[0].scatter(np.exp(g_1_lst), latencies_sigma/10**3, c='red', marker='o', alpha=.5, s=15)
	axes_task[0].set_ylim(*times_ylim)

	twin_ax0 = axes_task[0].twinx()
	twin_ax0.scatter(np.exp(g_1_lst), abruptness_sigma, c='blue', marker='o', alpha=.5, s=15)
	twin_ax0.set_yticks([])
	twin_ax0.set_ylim(*rates_ylim)

	acc_ax0 = axes_task[0].twinx()
	acc_ax0.set_yticks([])
	acc_ax0.scatter(np.exp(g_1_lst), sigma_acc[:,-1], c='black', marker='o', alpha=.5, s=15)
	acc_ax0.set_ylim(*acc_ylim)

	# Plot abruptness, latency and accuracy vs beta
	axes_task[1].set_xlabel(r'$\exp(g_2[P^*])$')
	axes_task[1].set_yticks([])
	axes_task[1].scatter(np.exp(g_2_lst), latencies_beta/10**3, c='red', marker='o', alpha=.5, s=15)
	axes_task[1].set_ylim(*times_ylim)

	twin_ax1 = axes_task[1].twinx()
	twin_ax1.set_ylabel(r'Abruptness ($\times 10^{-3}$)', color='blue')
	twin_ax1.scatter(np.exp(g_2_lst), abruptness_beta, c='blue', marker='o', alpha=.5, s=15)
	twin_ax1.tick_params(axis='y', colors='blue')
	twin_ax1.set_ylim(*rates_ylim)
	# rescale abruptness so that it looks nicer
	ticks_y = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale_rates))
	twin_ax1.yaxis.set_major_formatter(ticks_y)

	acc_ax1 = axes_task[1].twinx()
	acc_ax1.spines['right'].set_position(('axes', 1.45))
	acc_ax1.set_ylabel('Final Accuracy', color='black')
	acc_ax1.scatter(np.exp(g_2_lst), beta_acc[:,-1], c='black', marker='o', alpha=.5, s=15)
	acc_ax1.set_ylim(*acc_ylim)


	# save figure
	fig_task.savefig("fig6.pdf", format='pdf', dpi=500)

	# fitted curve latency vs abruptness (Figure 3c)
	fit_curve_f3 = f3(np.sort(abruptness_random), k0, k1)
	# computing r^2
	SS_res = ((latencies_random[np.argsort(abruptness_random)] - fit_curve_f3)**2).sum()
	SS_tot = ((latencies_random - latencies_random.mean())**2).sum()
	r_square_f3 = 1-SS_res/SS_tot

	# Make plot
	fig_rand, ax_rand = plt.subplots(figsize=(1.7, 1.7), layout="constrained")

	ax_rand.set_xlabel(r'Abruptness')
	ax_rand.set_ylabel(r'Latency ($\times 10^3$)')
	ax_rand.scatter(abruptness_random, latencies_random/10**3, c='green', marker='o', alpha=.5, s=15)
	ax_rand.plot(np.sort(abruptness_random), fit_curve_f3/10**3, 'g-')
	ax_rand.annotate(r"$\tau\propto \rho^{-1}$", (.3,.3), xycoords='axes fraction', c='green')

	fig_rand.savefig('fig4b.pdf', format='pdf', dpi=500)

	return k0,k1,r_square_f0,r_square_f1,r_square_f2, r_square_f3

def plot_MI():
	"""
	Plot the exponentiated MI over P(y|x) for discrete x=0,1 where P(x)=0.5.
	In this case, there is no volatility function and the prior is just
	exp(MI) where MI is the mutual information between X and Y. 

	This plot is a heatmap which shows that it is impossible to go from
	one high likelihood region to another without crossing a low-likelihood
	region. Thus, making the set of allowed hypotheses effectibely discrete.
	"""

	def h(p):
	    """Calculate binary entropy"""
	    p = np.asarray(p) # Ensure input is array for vectorized ops
	    # Use np.where to handle 0*log(0) = 0
	    term1 = np.where(p == 0, 0.0, p * np.log(p))
	    term2 = np.where(p == 1, 0.0, (1 - p) * np.log(1 - p))
	    # Entropy is non-negative, ensure result is >= 0
	    result = -(term1 + term2)
	    return np.maximum(0.0, result) # Clamp potential small negative due to precision

	# Define the function V(p0, p1) = exp(I[x;y])
	def V(p0, p1):
	    """Calculates exp(Mutual Information) for given P(y=1|x=0) and P(y=1|x=1)."""
	    p0 = np.asarray(p0)
	    p1 = np.asarray(p1)

	    # Calculate H[Y]
	    H_Y = h(0.5 * (p0 + p1))

	    # Calculate H[Y|X]
	    H_YcondX = 0.5 * h(p0) + 0.5 * h(p1)

	    # Calculate mutual information I[X;Y] = H[Y] - H[Y|X]
	    IMI = np.maximum(0.0, H_Y - H_YcondX)

	    return np.exp(IMI)

	# Create grid of coordinates
	num_points = 200 # Resolution of the grid
	p0_vals = np.linspace(0, 1, num_points)
	p1_vals = np.linspace(0, 1, num_points)
	P0, P1 = np.meshgrid(p0_vals, p1_vals)

	# Compute V, plot
	V_grid = V(P0, P1)
	fig, ax = plt.subplots(figsize=(2.2,1.7), layout="constrained")

	# Define custom colormap
	cmap_white_red = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])

	# display heatmap
	im = ax.imshow(V_grid,
	               origin='lower',
	               extent=[0, 1, 0, 1],
	               cmap=cmap_white_red,
	               vmin=1,
	               vmax=2,
	               aspect='equal'
	              )

	# set labels
	ax.set_xlabel(r"$P(y=1|x=0)$", labelpad=2)
	ax.set_ylabel(r"$P(y=1|x=1)$", labelpad=2)
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.tick_params(axis='both', which='major', labelsize=10)

	# Image border
	for spine in ax.spines.values():
	    spine.set_edgecolor('black')
	    spine.set_linewidth(2.0)

	# Colorbar
	cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
	cbar.set_label(r"$\exp(g_2[P])$", rotation=270, labelpad=15)
	# Set specific ticks on the colorbar
	cbar.set_ticks(np.linspace(1, 2, 6))
	cbar.ax.tick_params(labelsize=10)


	fig.savefig("fig7a.pdf", format='pdf', dpi=500)

def plot_population_curves():
	"""
	Given random parameter variation data, plot the various learning curves
	and a the population-averaged curve.
	"""



def plot_random_task(a, b, c, N=100, T=3000, p0=None, discourage=False):
	"""
	Run a single associative learning experiment with a preference-discouraging or random
	reward schedule. Plot change in distribution over time, with inset showing the final
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
	discourage : bool
		Whether or not to use the preference-discouraging data. If not, random
		samples will be used throughout. 
	"""

	T = int(T)
	# Generate fully random true distribution
	truth = 0.5 * np.ones((N,N,2))
	# Generate 2*T samples from the ground truth
	samples = gen_samples_cond(2*T,truth)
	if discourage:
		# Get results from learning with preference-discouraging
		t,ps,msg = get_dist(samples, a, b, c, p0=p0, N=N, T=T, pd=True)
	else:
		# Get results from learning with preference-discouraging
		t,ps,msg = get_dist(samples, a, b, c, p0=p0, N=N, T=T, pd=False)

	# average magnitude of derivative of P_t(y|x) over time
	dp_dt = np.abs(np.diff(ps, axis=0)).mean(axis=(1,2,3))

	fig, ax = plt.subplots(figsize=(2.7,1.8), layout="constrained")
	
	# Plot change in P over time
	ax.plot(t[1:], dp_dt*10000, 'k')
	ax.set_xlabel('Time')
	ax.set_ylabel(r'Update Speed ($\times 10^{-4}$)')

	# Create inset with final dist
	inset_bounds = [0.65, 0.65, 0.3, 0.3] # inset location, size
	axins = ax.inset_axes(inset_bounds)
	axins.annotate(r"$\hat P_T$", (-0.15,-0.45), xycoords='axes fraction')

	# show final output
	axins.imshow(ps[-1,:,:,1], cmap='coolwarm', vmin=0, vmax=1, origin='lower')
	axins.set_xticks([])
	axins.set_yticks([])

	# Draw connecting lines for inset
	# connection coords for bottom right
	xyA = (0.95, 0.075)
	# connection coords for bottom right and top rightof inset
	xyB_corner1 = (1.0, 1.0)
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

	# show final output
	axins.imshow(ps[-1,:,:,1], cmap='coolwarm', vmin=0, vmax=1, origin='lower')
	axins.set_xticks([])
	axins.set_yticks([])

	# Make second inset
	inset_bounds2 = [0.4, 0.3, 0.3, 0.3] # inset location, size
	axins2= ax.inset_axes(inset_bounds2)
	axins2.annotate(r"$\hat P_{.75T}$", (-0.15,-0.45), xycoords='axes fraction')


	# Draw connecting lines for inset
	# connection coords for bottom right
	xyA = (0.72, 0.075)
	# connection coords for bottom left and top left of inset
	xyB_corner1 = (1.0,0.0)
	xyB_corner2 = (1.0,1.0)

	# Create first line to bottom-left of inset
	con3 = ConnectionPatch(
	    xyA=xyA, coordsA=ax.transAxes,
	    xyB=xyB_corner1, coordsB=axins2.transAxes,
	    arrowstyle="-", 
	    linestyle="--",
	    color="gray",
	    linewidth=1.0
	)

	# Create second line to bottom right of inset
	con4 = ConnectionPatch(
	    xyA=xyA, coordsA=ax.transAxes,
	    xyB=xyB_corner2, coordsB=axins2.transAxes,
	    arrowstyle="-",
	    linestyle="--",
	    color="gray",
	    linewidth=1.0
	)

	fig.add_artist(con3)
	fig.add_artist(con4)

	# show final output
	axins2.imshow(ps[int(.75*T),:,:,1], cmap='coolwarm', vmin=0, vmax=1, origin='lower')
	axins2.set_xticks([])
	axins2.set_yticks([])

	ax.set_clip_on(False)
	axins.set_clip_on(False)
	axins2.set_clip_on(False)

	if discourage==False:
		fig.savefig('fig7b.pdf', format='pdf', dpi=500)
	else:
		fig.savefig('fig7c.pdf', format='pdf', dpi=500)

	# return average change in probs at end of curve
	delta_p = np.abs(ps[int(T*0.9)] - ps[-1]).mean()

	return delta_p



