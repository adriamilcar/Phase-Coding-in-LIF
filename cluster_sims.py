import numpy as np
from scipy.stats import norm
import pickle
import itertools


def grid_freqNoise_exp(idx):

	def get_automatic_range(R_m, V_th, tau_m, omega, I_osc, range_frac=.9):
		A = 1/np.sqrt(1 + (tau_m*omega)**2)
		T = (2*np.pi)/omega
		expon = np.exp(-T/tau_m)
		frac_denom = (1 - expon)*R_m*I_osc*A
		
		I_min = (V_th/frac_denom - 1)*I_osc*A
		I_max = (V_th/frac_denom + 1)*I_osc*A
		
		corr_frac = (1 - range_frac) * (I_max - I_min) / 2
		I_min += corr_frac
		I_max -= corr_frac
		
		return I_min, I_max

	def get_snr_corr_factor(tau_m, f):
		delta_Is = (1 - np.exp(-1/tau_m)) / (1 - np.exp(-1/(f*tau_m)))
		delta_Iosc = np.sqrt(1 + (tau_m*2*np.pi*f)**2) / np.sqrt(1 + (tau_m*2*np.pi)**2)
		snr = delta_Is * delta_Iosc    # delta_Is + delta_Iosc ??
		return snr

	def compute_Iosc(v_osc_amp, R_m, tau_m, omega):
		I_osc = abs(v_osc_amp) * np.sqrt((tau_m*omega)**2 + 1) / R_m
		return I_osc

	def compute_mean_phi(R_m, V_th, tau_m, omega, I_s, I_osc):
		A = 1/np.sqrt(1 + (tau_m*omega)**2)
		ph = -np.arctan(tau_m*omega) - np.pi
		m = I_s/(A*I_osc)
		T = (2*np.pi)/omega
		b = V_th/(R_m*I_osc*A*(1 - np.exp(-T/tau_m)))
		phi = np.nan_to_num(-np.arccos(b - m) - ph)
		return phi

	def simulate_neurons(R_m, V_th, eta, tau_m, omega, I_s, v_osc_amp, f, M, dt, t, n_neurons):
		snr = get_snr_corr_factor(tau_m, f)
		sigma_W = eta * snr * V_th / np.sqrt(tau_m)
		I_osc = compute_Iosc(v_osc_amp, R_m, tau_m, omega)
		phi_0 = compute_mean_phi(R_m, V_th, tau_m, omega, I_s, I_osc)
		
		V = np.zeros(n_neurons)
		first_spike_phase = np.empty(n_neurons, dtype='object')
		first_spike_phase[:] = None
		has_spiked = np.zeros(n_neurons, dtype=bool)
		for i in range(len(t)):
			xi = np.random.normal(0, 1/np.sqrt(dt), n_neurons)
			I_theta = I_osc * np.cos(omega * t[i] - np.pi + phi_0)
			dV_dt = (-V + R_m * I_theta + R_m * I_s + tau_m * sigma_W * xi) / tau_m
			V += dV_dt * dt
		
			spiked = V >= V_th
			if np.any(spiked):
				V[spiked] = 0 
				if (omega * t[i] - np.pi + phi_0) > np.pi:
					mask = spiked & (first_spike_phase == None)
					first_spike_phase[mask] = (omega * t[i] + phi_0) % (2*np.pi)

					if np.all(first_spike_phase != None):
						break
					
		return first_spike_phase

	def get_distr_empirical(R_m, V_th, eta, tau_m, v_osc_amp, f, M, dt, t, num_trials, range_frac):
		omega = 2*np.pi*f
		I_osc = compute_Iosc(v_osc_amp, R_m, tau_m, omega)
		I_min, I_max = get_automatic_range(R_m, V_th, tau_m, omega, I_osc, range_frac=range_frac)
		Is_range = np.linspace(I_min, I_max, M)
		
		means = []
		variances = []
		all_first_spike_phases = []
		for I_s in Is_range:
			first_spike_phases = simulate_neurons(R_m, V_th, eta, tau_m, omega, I_s, v_osc_amp, f, M, dt, t, num_trials)
			first_spike_phases = np.where(first_spike_phases == None, np.nan, first_spike_phases)
			all_first_spike_phases.append(first_spike_phases)  
			means.append( np.nanmean(first_spike_phases) )
			variances.append( np.nanvar(first_spike_phases) )
			
		return np.array(means), np.array(variances), all_first_spike_phases, Is_range

	def numerical_mi(means, variances, n_samples=int(1e4)):
		samples = []
		for mean, var in zip(means, variances):
			samples.extend(norm.rvs(loc=mean, scale=np.sqrt(var), size=n_samples))
		histogram, _ = np.histogram(samples, bins='auto', density=True)
		histogram = histogram + np.finfo(float).eps
		H_Y = -np.sum(histogram * np.log(histogram) * np.diff(np.linspace(np.min(samples), np.max(samples), len(histogram) + 1)))

		H_Y_given_X = np.mean([0.5 * np.log(2 * np.pi * np.e * var) for var in variances])

		return H_Y - H_Y_given_X

	def get_R(mis, fs, tau_x, norm=True):
		corr_fact_sampling = (1 - np.exp(-(1/(fs*tau_x))))
		info_rate = mis * fs[np.newaxis, :] * corr_fact_sampling[np.newaxis, :]
		
		if norm:
			info_rate_norm = np.zeros_like(info_rate)  
			for index, row in enumerate(info_rate):
				info_rate_norm[index, :] = row/np.max(row)
			info_rate = info_rate_norm
			
		return info_rate



	# Params
	N = 1                  # number of neurons
	R_m = 142 * 1e6        # MOmh
	V_th = 15 * 1e-3       # mV
	tau_m = 24 * 1e-3      # ms
	v_osc_amp = 5 * 1e-3   # mV
	M = 10                  # number input levels
	range_frac = 0.75

	# Get noise and f from idx
	noise_lims = [0.001, 0.5]
	f_lims = [1, 100]
	res = 200

	noise_fracs = np.linspace(noise_lims[0], noise_lims[1], res)
	fs = np.linspace(f_lims[0], f_lims[1], res)

	experiments = list(itertools.product(noise_fracs, fs))
	num_experiments = len(list(experiments))

	eta, f = experiments[idx]

	# Simulation parameters
	dt = 1e-4            # Time step for numerical integration
	num_trials = 5000    # Number of trials


	## SIMULATION
	t_end = 2 / f
	t = np.arange(0, t_end, dt)
	means, variances, all_phis, Is_range = get_distr_empirical(R_m, V_th, eta, tau_m, v_osc_amp, f, M, dt, t, num_trials, range_frac)
	MI = numerical_mi(means, variances, n_samples=int(2e4))


	## Save results
	results_dict = {}
	results_dict["MI"] = MI

	filename = 'Data/' + 'experimental_' + '{0:03d}'.format(int(idx/res)) + '_' + '{0:03d}'.format(int(idx%res)) + ".pickle"

	with open(filename, 'wb') as handle:
		pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
