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

	if args.fig1b:
		# Generate sloppiness experiment figure, collect data if necessary.
		if args.collect_data_sloppy:
			print("Generating Figure 1b and collecting data.")
			scale = px_vs_constraint(sigma=30, beta=50, p_high=1.0, N=100, nsamples=1000, collect_data=True)
			print(r"Volatility Scale ($k$): {}".format(scale))
		else:
			print("Generating Figure 1b, using pre-collected data.")
			try:
				scale = px_vs_constraint(sigma=30, beta=50, p_high=1.0, N=100, nsamples=1000, collect_data=False)
				print(r"Volatility Scale ($k$): {}".format(scale))
			except FileNotFoundError:
				print("Error: Data not found for Figure 1b. Run with --collect-data-sloppy first.")
			except Exception as e:
				print(f"An error occurred generating Figure 1b: {e}")

	if args.fig2b:
		print("Generating Figure 2b.")
		try:
			single_plot(a=0.001, b=0.01, c=0.01, sigma=20,beta=50,N=100, T=2000)
		except Exception as e:
			print(f"An error occurred generating Figure 2b: {e}")

	# collect parameter variation data
	if args.collect_param_data:
		print("Collecting data for parameter variation (Figure 3).")
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

		try:
			collect_data_param_variation(a_lst, b_lst, c_lst, sigma_lst, beta_lst, N=100, T=3000)
		except Exception as e:
			print(f"An error occurred during parameter variation data collection: {e}")

	# Generating Figure 3, requires parameter variation data
	if args.fig3:
		print("Generating Figure 3 (requires collected parameter data).")
		try:
			k0, k1 = plot_param_variation()
			print('Fit Results:')
			print(r'k0 = {:.5e}'.format(k0))
			print('k1 = {:.5}'.format(k1))
		except FileNotFoundError:
			print("Error: Parameter variation data file not found. Run with --collect-param-data first.")
		except Exception as e:
			print(f"An error occurred generating Figure 3: {e}")

	# Generating Figure 4a: plot exponentiated MI
	if args.fig4a:
		print("Generating Figure 4a")
		try:
			plot_MI()
		except Exception as e:
			print(f"An error occurred generating Figure 4a: {e}")

	# Generating Figure 4b: plot dp/dt with discouraging reward schedule
	if args.fig4b:
		print("Generating Figure 4b")
		try:
			delta_p = plot_random_task(a=0.001, b=0.01, c=0.01, N=100, T=4000)
			print(r'Average $|\Delta \hat P|$ from .75 completed to end is {}'.format(delta_p))
		except Exception as e:
			print(f"An error occurred generating Figure 4b: {e}")

	print("Script Finished")
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate figures and collect data for the paper.")

	# Flags to enable figure generation (default is False)
	parser.add_argument('--fig1b', action='store_true', 
    	help='Generate Figure 1b (Sloppiness/Prior Weight). Requires --collect-data-sloppy first if data is not present.')
	parser.add_argument('--fig2b', action='store_true', 
    	help='Generate Figure 2b (Simple Learning Curve).')
	parser.add_argument('--fig3', action='store_true', 
    	help='Generate Figure 3 (Parameter Variation Scaling). Requires --collect-param-data first if data is not present.')
	parser.add_argument('--fig4a', action='store_true', 
    	help='Generate Figure 4a (Exponentiated Mutual Information).')
	parser.add_argument('--fig4b', action='store_true', 
    	help='Generate Figure 4b (Persistence/Random Task).')

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
		args.fig1b = True
		args.fig2b = True
		args.fig3 = True
		args.fig4a = True
		args.fig4b = True
		args.collect_param_data = True
		args.collect_data_sloppy = True

	# run main
	main(args)