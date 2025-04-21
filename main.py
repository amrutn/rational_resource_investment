from plots import *


# Generate all the figures.

def main(fig1b = True, collect_param_data=True, fig2=True, fig3a=True, fig3b=True, fig4=True, collect_px_data=True):
	# Figure 1b
	if fig1b:
		single_plot(a=0.001, b=0.01, c=0.01, sigma=20,beta=50,N=100, T=2000)

	# collect param_variation data
	if collect_param_data:
		# a varies from .00001 to .1, anchor .001
		a_lst = np.exp(np.linspace(np.log(.00001),np.log(.1),25))
		a_lst = np.insert(a_lst, 0, .001)

		# b varies from .0001 to 1.5, anchor .01
		b_lst = np.exp(np.linspace(np.log(.0001),np.log(1.5),25))
		b_lst = np.insert(b_lst, 0, .01)

		# c varies from 0.0075 to 0.025, anchor 0.01
		c_lst = np.linspace(.001,0.02,20)
		c_lst = np.insert(c_lst, 0, .01)

		# sigma varies from 1 to 25, anchor 20
		sigma_lst = np.linspace(1,25,25)
		sigma_lst = np.insert(sigma_lst, 0, 20)

		# beta varies from 1 to 10, anchor 10
		beta_lst = np.exp(np.linspace(np.log(0.05),np.log(50),25))
		beta_lst = np.insert(beta_lst, 0, 50)

		collect_data_param_variation(a_lst, b_lst, c_lst, sigma_lst, beta_lst, N=100, T=3000)

	# Figure 2a,b
	if fig2:
		k0, k1 = plot_param_variation()
		print(r'k0 = {:.5e}'.format(k0), flush=True)
		print('k1 = {:.5}'.format(k1), flush=True)

	# Figure 3a
	if fig3a:
		plot_prior()

	# Figure 3b
	if fig3b:
		delta_p = plot_random_task(a=0.001, b=0.01, c=0.01, N=100, T=4000)
		print(r'Average $|\Delta \hat P|$ from .75 completed to end is {}'.format(delta_p), flush=True)

	# Figure 4
	if fig4:
		px_vs_constraint(sigma=30, beta=50, p_high=1.0, N=100, nsamples=1000, collect_data=collect_px_data)
	return

main(collect_param_data=False, collect_px_data=True, fig1b=False,fig2=False, fig3a=False, fig3b=False)