from plots import *
import argparse

# Generate all figures
# --- Example Usage ---
#
# Collect data for Figure 3:
# python main.py --collect-param-data
#
# Generate Figure 3 (assuming data was collected previously):
# python main.py --fig3
#
# Generate Figures 2b and 4a:
# python main.py --fig2b --fig4a
#
# Collect data for Figure 1b and generate it:
# python main.py --collect-px-data --fig1b
#
# Run everything:
# python main.py --all
#
# Run just Figure 4b (as in the original example call):
# python main.py --fig4b

# Note, the seed is preset to 42

def main(args):
	# Generate figures and data based on parsed arguments.

	if args.fig2b:
		# Generate sloppiness experiment figure, collect data if necessary.
		if args.collect_data_sloppy:
			print("Generating Figure 2b and collecting data.")
			scale = px_vs_constraint(sigma=30, beta=50, p_high=1.0, N=100, nsamples=1000, collect_data=True)
			print(r"Volatility Scale ($k$): {}".format(scale))
		else:
			print("Generating Figure 2b, using pre-collected data.")
			try:
				scale = px_vs_constraint(sigma=30, beta=50, p_high=1.0, N=100, nsamples=1000, collect_data=False)
				print(r"Volatility Scale ($k$): {}".format(scale))
			except FileNotFoundError:
				print("Error: Data not found for Figure 2b. Run with --collect-data-sloppy first.")
			except Exception as e:
				print(f"An error occurred generating Figure 2b: {e}")

	if args.fig3b:
		print("Generating Figure 3b.")
		try:
			single_plot(a=0.001, b=0.01, c=0.01, sigma=20,beta=50,N=100, T=2000)
		except Exception as e:
			print(f"An error occurred generating Figure 3b: {e}")

	# collect parameter variation data
	if args.collect_param_data:
		print("Collecting data for parameter variation (Figure 3 and 4).")
		# a varies from .00001 to .1, anchor .001
		a_lst = np.exp(np.linspace(np.log(.00001),np.log(.1),25))
		a_lst = np.insert(a_lst, 0, .001)

		# b varies from .0001 to 1.5, anchor .01
		b_lst = np.exp(np.linspace(np.log(.0001),np.log(1.5),25))
		b_lst = np.insert(b_lst, 0, .01)

		# c varies from 0.0075 to 0.02, anchor 0.01
		c_lst = np.linspace(.0075,0.02,20)
		c_lst = np.insert(c_lst, 0, .01)

		# sigma varies from 1 to 25, anchor 20
		sigma_lst = np.linspace(1,25,25)
		sigma_lst = np.insert(sigma_lst, 0, 20)

		# beta varies from 1 to 10, anchor 10
		beta_lst = np.exp(np.linspace(np.log(0.05),np.log(50),25))
		beta_lst = np.insert(beta_lst, 0, 50)

		try:
			collect_data_param_variation(a_lst, b_lst, c_lst, sigma_lst, beta_lst, N=100, T=3000)
		except Exception as e:
			print(f"An error occurred during parameter variation data collection: {e}")

	# Generating Figure 4 and 6, requires parameter variation data
	if args.fig4_6:
		print("Generating Figures 4 and 6 (requires collected parameter data).")
		try:
			k0, k1,r_square_f0,r_square_f1,r_square_f2, r_square_f3 = plot_param_variation()
			print('Fit Results:')
			print(r'k0 = {:.5e}'.format(k0))
			print('k1 = {:.5}'.format(k1))
			print(r'$R^2$ latencies vary $a$ = {:.5}'.format(r_square_f0))
			print(r'$R^2$ abruptness vary $c$ = {:.5}'.format(r_square_f1))
			print(r'$R^2$ latencies vary $c$ = {:.5}'.format(r_square_f2))
			print(r'$R^2$ latencies vs abruptness = {:.5}'.format(r_square_f3))
		except FileNotFoundError:
			print("Error: Parameter variation data file not found. Run with --collect-param-data first.")
		except Exception as e:
			print(f"An error occurred generating Figures 4 and 6: {e}")

	if args.fig5:
		print("Generating synthetic replica of Gallistel's plot, Figure 5")
		try:
			plot_random_curves()
		except FileNotFoundError:
			print("Error: Random parameter variation data file not found. Run with --collect-param-data first.")
		except Exception as e:
			print(f"An error occurred generating synthetic Gallistel replica: {e}")

	# Generating Figure 7a: plot exponentiated MI
	if args.fig7a:
		print("Generating Figure 7a")
		try:
			plot_MI()
		except Exception as e:
			print(f"An error occurred generating Figure 7a: {e}")

	# Generating Figure 7b and 7c: plot dp/dt with uniform and discouraging reward schedule
	if args.fig7bc:
		print("Generating Figure 7b")
		try:
			delta_p = plot_random_task(a=0.001, b=0.01, c=0.01, N=100, T=4000, discourage=False)
			print(r'Average $|\Delta \hat P|$ from .75 completed to end is {}'.format(delta_p))
		except Exception as e:
			print(f"An error occurred generating Figure 7b: {e}")

		print("Generating Figure 7c")
		try:
			delta_p = plot_random_task(a=0.001, b=0.01, c=0.01, N=100, T=4000, discourage=True)
			print(r'Average $|\Delta \hat P|$ from .75 completed to end is {}'.format(delta_p))
		except Exception as e:
			print(f"An error occurred generating Figure 7c: {e}")

	# Generate supplemental figure for the incorrect input marginal
	if args.fig8:
		print("Generating Figure 8.")
		try:
			single_plot_wrong_inp_marg(a=0.001, b=0.001, c=0.01, sigma=10,beta=50,N=100, T=2000)
		except Exception as e:
			print(f"An error occurred generating Figure 8: {e}")

	# Generate supplemental figure for a biased true distribution
	if args.fig9:
		print("Generating Figure 9.")
		try:
			single_plot_biased_truth(a=0.001, b=0.01, c=0.01, sigma=20,beta=50,N=100, T=2000)
		except Exception as e:
			print(f"An error occurred generating Figure 9: {e}")
	if args.fig10:
		print("Generating Figure 10.")
		try:
			# plot showing the sudden insight
			single_plot_nolog(a=0.001, b=0.01, c=0.01, sigma=20,beta=50,N=100, T=2000)
			# plot showing the superstition
			plot_random_task(a=0.001, b=0.01, c=0.01, N=100, T=4000, discourage=False)
			# plot showing the persistent superstition
			plot_random_task(a=0.001, b=0.01, c=0.01, N=100, T=4000, discourage=True)
		except Exception as e:
			print(f"An error occurred generating Figure 10: {e}")

	print("Script Finished")
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate figures and collect data for the paper.")

	# Flags to enable figure generation (default is False)
	parser.add_argument('--fig2b', action='store_true', 
    	help='Generate Figure 2b (Sloppiness/Prior Weight). Requires --collect-data-sloppy first if data is not present.')
	parser.add_argument('--fig3b', action='store_true', 
    	help='Generate Figure 3b (Simple Learning Curve).')
	parser.add_argument('--fig4-6', action='store_true', 
    	help='Generate Figure 4 and 6 (Parameter Variation Scaling). Requires --collect-param-data first if data is not present.')
	parser.add_argument('--fig5', action='store_true', 
    	help='Generate Figure 5. Requires --collect-param-data first if data is not present.')
	parser.add_argument('--fig7a', action='store_true', 
    	help='Generate Figure 7a (Exponentiated Mutual Information).')
	parser.add_argument('--fig7bc', action='store_true', 
    	help='Generate Figure 7b and 7c (Persistence/Random Task).')
	parser.add_argument('--fig8', action='store_true', 
    	help='Generate Figure 8 (Learning curve for incorrect input marginal).')
	parser.add_argument('--fig9', action='store_true', 
    	help='Generate Figure 9 (Learning curve for biased true distribution).')
	parser.add_argument('--fig10', action='store_true', 
    	help='Generate Figure 10 (Learning curve for smoothness without log).')

	# Flags to enable data collection (default is False)
	parser.add_argument('--collect-param-data', action='store_true', 
    	help='Run data collection for Figure 3 (Parameter Variation). This can be time-consuming.')
	parser.add_argument('--collect-data-sloppy', action='store_true', 
    	help='Run data collection for Figure 1b (Sloppiness/Prior Weight). This can be time-consuming.')

	# Flag to run everything (figures and data collection)
	parser.add_argument('--all', action='store_true', help='Run all data collection steps and generate all figures.')

	args = parser.parse_args()

	# If --all is specified, set all flags to True
	if args.all:
		args.fig2b = True
		args.fig3b = True
		args.fig4_6 = True
		args.fig5 = True
		args.fig7a = True
		args.fig7bc = True
		args.collect_param_data = True
		args.collect_data_sloppy = True

	# run main
	main(args)