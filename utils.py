import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
import pickle



def Gaussian_pdf(mu, sigma, stds=4):
    x = np.linspace(mu - stds*sigma, mu + stds*sigma, 100)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return x, y


def truncated_Gaussian_moments(mu, var, a=0):
    """Compute the mean and variance of a truncated normal distribution."""
    sigma = np.sqrt(var)
    alpha = (a - mu) / sigma
    
    Z = 1 - norm.cdf(alpha)
    phi_alpha = norm.pdf(alpha)
    
    mu_t = mu + sigma * (phi_alpha / Z)
    var_t = var * (1 + (alpha * phi_alpha / Z) - (phi_alpha / Z) ** 2)
    
    return mu_t, var_t


def sigma_w_to_eta(sigma_w, V_th, tau_m, f):
    """Convert sigma_w to eta."""
    return (sigma_w * np.sqrt(tau_m)) / V_th


def compute_mean_phi(R_m, V_th, tau_m, omega, I_s, I_osc):
    """Compute the expected phase of firing."""
    A = 1 / np.sqrt(1 + (tau_m * omega)**2)
    T = (2 * np.pi) / omega
    varphi = -np.arctan(omega * tau_m) 
    numerator = R_m * I_s * (1 - np.exp(-T / tau_m)) - V_th
    denominator = R_m * I_osc * A * (1 - np.exp(-T / tau_m))
    phi_i = np.arccos(numerator / denominator) - varphi
    return phi_i


def compute_var_phi(R_m, V_th, eta, tau_m, omega, I_s, I_osc, f, N):
    """Compute the variance in phase of firing."""
    phi_Is = compute_mean_phi(R_m, V_th, tau_m, omega, I_s, I_osc)
    exp_term = np.exp(-2 / (f * tau_m))
    alpha = I_s/I_s_ref
    numerator = omega**2 * alpha**2 * eta**2 * V_th**2 * (1 - exp_term) * tau_m**2
    denominator = 2 * (-V_th + R_m * I_s - R_m * I_osc * np.cos(phi_Is))**2
    var_phi = numerator / denominator
    return var_phi


def compute_var_phi_fromPRC_1(R_m, V_th, eta, tau_m, omega, I_s, I_osc, f, N):
    """Compute the variance in phase of firing."""
    T = 1 / f
    A = 1 / np.sqrt(1 + (tau_m * omega) ** 2)
    varphi = -np.arctan(omega * tau_m)
    phi_Is = compute_mean_phi(R_m, V_th, tau_m, omega, I_s, I_osc)
    alpha = I_s/I_s_ref
    numerator = alpha**2 * eta**2 * V_th**2 * (1 - np.exp(-2*T / tau_m))
    denominator = 2 * R_m**2 * I_osc**2 * A**2 * (1 - np.exp(-T / tau_m))**2 * np.sin(phi_Is + varphi)**2
    var_phi_Is = numerator / denominator
    return var_phi_Is


def compute_var_phi_fromPRC_2(R_m, V_th, eta, tau_m, omega, I_s, I_osc, f, N):
    """Compute the variance in phase of firing."""
    T = 1 / f
    A = 1 / np.sqrt(1 + (tau_m * omega) ** 2)
    eff_osc_amp = R_m * I_osc * A * (1 - np.exp(-T / tau_m))  # effective oscillation amplitude
    eff_stim_current = R_m * I_s * (1 - np.exp(-T / tau_m))   # effective stimulus current
    alpha = I_s/I_s_ref
    numerator = alpha**2 * eta**2 * V_th**2 * (1 - np.exp(-2 * T / tau_m))
    denominator = 2 * (eff_osc_amp**2 - (V_th - eff_stim_current)**2)
    var_phi = numerator / denominator
    return var_phi


def get_distr(R_m, V_th, eta, tau_m, I_osc, f, M, range_frac, N):
    omega = 2 * np.pi * f
    I_min, I_max = get_automatic_range(R_m, V_th, tau_m, omega, I_osc, range_frac)
    Is_range = np.linspace(I_min, I_max, M)
    means = [compute_mean_phi(R_m, V_th, tau_m, omega, I_s, I_osc) for I_s in Is_range]
    variances = [compute_var_phi(R_m, V_th, eta, tau_m, omega, I_s, I_osc, f, N) for I_s in Is_range]
    return np.array(means), np.array(variances), Is_range


def get_automatic_range(R_m, V_th, tau_m, omega, I_osc, range_frac=.9):
    A = 1 / np.sqrt(1 + (tau_m * omega)**2)
    T = (2 * np.pi) / omega
    expon = np.exp(-T / tau_m)
    frac_denom = (1 - expon) * R_m * I_osc * A
    
    I_min = (V_th / frac_denom - 1) * I_osc * A
    I_max = (V_th / frac_denom + 1) * I_osc * A
    
    corr_frac = (1 - range_frac) * (I_max - I_min) / 2
    I_min += corr_frac
    I_max -= corr_frac
    
    return I_min, I_max


def approx_mi(means, variances):
	K = np.log(2 * np.pi * np.exp(1)) / 2
	mi = K * (np.log(np.mean(variances) + np.var(means)) - np.mean(np.log(variances)))
	return mi


def numerical_mi(means, variances, n_samples=int(1e4)):
    samples = []
    for mean, var in zip(means, variances):
        samples.extend(norm.rvs(loc=mean, scale=np.sqrt(var), size=n_samples))
    histogram, _ = np.histogram(samples, bins='auto', density=True)
    histogram = histogram + np.finfo(float).eps
    H_Y = -np.sum(histogram * np.log(histogram) * np.diff(np.linspace(np.min(samples), np.max(samples), len(histogram) + 1)))

    H_Y_given_X = np.mean([0.5 * np.log(2 * np.pi * np.e * var) for var in variances])

    return H_Y - H_Y_given_X


def get_R(mis, fs, tau_s, corr=True, norm=True):
    info_rate = mis * fs[np.newaxis, :]
    
    if corr:
        corr_fact_sampling = (1 - np.exp(-(1/(fs*tau_s))))
        info_rate = info_rate * corr_fact_sampling[np.newaxis, :]
    
    if norm:
        info_rate_norm = np.zeros_like(info_rate)  
        for index, row in enumerate(info_rate):
            info_rate_norm[index, :] = row/np.max(row)
        info_rate = info_rate_norm
        
    return info_rate


def simulate_neurons(R_m, V_th, eta, tau_m, omega, I_s, I_s_ref, I_osc, f, M, dt, t, n_neurons, store_trajectories=False):
    alpha = I_s / I_s_ref
    sigma_W = eta * alpha * V_th / np.sqrt(tau_m)
    phi_0 = compute_mean_phi(R_m, V_th, tau_m, omega, I_s, I_osc)
    
    T = 1 / f
    A = 1 / np.sqrt(1 + (tau_m * omega)**2)
    
    V = np.zeros(n_neurons)
    first_spike_phase = np.empty(n_neurons, dtype='object')
    first_spike_phase[:] = None
    first_spike_times = np.full(n_neurons, np.nan)
    has_spiked = np.zeros(n_neurons, dtype=bool)
    
    if store_trajectories:
        voltage_trajectories = np.zeros((len(t), n_neurons)) * np.nan
    
    for i in range(len(t)):
        xi = np.random.normal(0, 1/np.sqrt(dt), n_neurons)
        I_theta = I_osc * np.cos(omega * t[i] + phi_0)
        dV_dt = (-V + R_m * (I_s - I_theta) + tau_m * sigma_W * xi) / tau_m
        V += dV_dt * dt
        
        if store_trajectories:
            voltage_trajectories[i, :] = V
            voltage_trajectories[i, has_spiked] = np.nan
        
        spiked = V >= V_th
        if np.any(spiked):
            V[spiked] = 0 
            if (omega * t[i] - np.pi + phi_0) > np.pi:
                mask = spiked & (first_spike_phase == None)
                first_spike_phase[mask] = (omega * t[i] + phi_0) % (2 * np.pi)
                first_spike_times[mask] = t[i]
                has_spiked[spiked] = 1
                if np.all(first_spike_phase != None):
                    break
    
    if store_trajectories:
        return first_spike_phase, voltage_trajectories, first_spike_times, phi_0
    else:
        return first_spike_phase


def get_distr_empirical(R_m, V_th, eta, tau_m, I_osc, f, M, dt, t, num_trials, range_frac, store_trajectories=False):
    omega = 2 * np.pi * f
    I_min, I_max = get_automatic_range(R_m, V_th, tau_m, omega, I_osc, range_frac)
    Is_range = np.linspace(I_min, I_max, M)

    I_s_ref, _ = get_automatic_range(R_m, V_th, tau_m, 2*np.pi*1, I_osc, range_frac)
    
    means = []
    variances = []
    all_first_spike_phases = []
    all_voltages = []
    all_first_spike_times = []
    all_phi_0 = []
    
    for I_s in Is_range:
        if store_trajectories:
            first_spike_phases, voltage_trajectories, first_spike_times, phi_0 = simulate_neurons(R_m, V_th, eta, tau_m, omega, I_s, I_s_ref, I_osc, f, M, dt, t, num_trials, store_trajectories)
            all_voltages.append(voltage_trajectories)
            all_first_spike_times.append(first_spike_times)
            all_phi_0.append(phi_0)
        else:
            first_spike_phases = simulate_neurons(R_m, V_th, eta, tau_m, omega, I_s, I_osc, f, M, dt, t, num_trials)
        
        first_spike_phases = np.where(first_spike_phases == None, np.nan, first_spike_phases)
        all_first_spike_phases.append(first_spike_phases)
        means.append(np.nanmean(first_spike_phases))
        variances.append(np.nanvar(first_spike_phases))
    
    if store_trajectories:
        return np.array(means), np.array(variances), all_first_spike_phases, Is_range, all_voltages, all_first_spike_times, all_phi_0
    else:
        return np.array(means), np.array(variances), all_first_spike_phases, Is_range


def amplitude_mem_potential(R_m, I_osc, tau_m, f):
    A = 1 / np.sqrt(1 + (tau_m * 2 * np.pi * f)**2)
    amplitude = R_m * I_osc * A
    return amplitude


def roll_with_nan(arr, shift):
    result = np.full_like(arr, np.nan)
    if shift > 0:
        result[shift:] = arr[:-shift]
    elif shift < 0:
        result[:shift] = arr[-shift:]
    else:
        result = arr
    return result
