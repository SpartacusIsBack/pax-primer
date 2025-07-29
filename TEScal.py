import mass
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.odr import ODR, Model, RealData

main_e = 59.5409e3 # eV, energy value of the main peak

def gaussian(x, a, mu, sigma, c):
    """Gaussian function with offset."""
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def fit_fwhm(x, y, plot=True, in_energy=False, return_sig=False):
    """
    Fit a Gaussian to the given spectrum and return the FWHM.

    Parameters:
    - x: array-like, bin centers (e.g., energy)
    - y: array-like, counts
    - plot: bool, if True, plot the fit

    Returns:
    - fwhm: Full Width at Half Maximum
    - popt: optimal parameters of the fit (a, mu, sigma, c)
    """
    # Initial guess: amplitude, mean, std, offset
    a0 = np.max(y) - np.min(y)
    mu0 = x[np.argmax(y)]
    sigma0 = (x[-1] - x[0]) / 10
    c0 = np.min(y)
    p0 = [a0, mu0, sigma0, c0]
    
    y_err = np.sqrt(y)  # Poisson errors (σ = √counts)
    y_err[y == 0] = 1   # Handle empty bins
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=y_err, p0=p0, absolute_sigma=True)
    except RuntimeError:
        print("Gaussian fit did not converge.")
        return None, None

    a, mu, sigma, c = popt
    fwhm = 2.3548 * abs(sigma)  # FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.3548*sigma

    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(x, y, 'b.', label='Data')
        xfit = np.linspace(np.min(x), np.max(x), 500)
        plt.plot(xfit, gaussian(xfit, *popt), 'r-', label='Gaussian fit')
        plt.axvline(mu, color='g', linestyle='--', label=f'Peak center {mu:.2f} PH')
        plt.axvspan(mu - fwhm/2, mu + fwhm/2, color='orange', alpha=0.2, label='FWHM')
        plt.ylabel('Counts')
        if in_energy:
            plt.xlabel('Energy (ev)')
            plt.title(f'Gaussian Fit: FWHM = {fwhm:.2f} ev')
        else:
            plt.xlabel('PH values')
            plt.title(f'Gaussian Fit: FWHM = {fwhm:.2f} PH')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if return_sig:
        sigmas = np.sqrt(np.diag(pcov))
        return fwhm, popt, sigmas
    else :
        return fwhm, popt


def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2, c):
    """
    Sum of two Gaussians plus constant background.
    """
    g1 = a1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    g2 = a2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    return g1 + g2 + c

def fit_double_gaussian(x, y, plot=True, in_energy=False, return_sig = False):
    """
    Fit the sum of two Gaussians to the given spectrum and return the FWHMs.

    Parameters:
    - x: array-like, bin centers (e.g., energy)
    - y: array-like, counts
    - plot: bool, if True, plot the fit

    Returns:
    - fwhms: tuple of FWHMs for both Gaussians
    - popt: optimal parameters of the fit (a1, mu1, sigma1, a2, mu2, sigma2, c)
    """
    # Initial guess: two peaks at the highest and second-highest bins
    idx1 = np.argmax(y)
    y_temp = y.copy()
    y_temp[idx1-7:idx1+8] = 0
    idx2 = np.argmax(y_temp)
    a1 = y[idx1] - np.min(y)
    mu1 = x[idx1]
    sigma1 = (x[-1] - x[0]) / 20
    a2 = a1/2
    mu2 = x[idx2]
    sigma2 = sigma1
    c0 = np.min(y)
    p0 = [a1, mu1, sigma1, a2, mu2, sigma2, c0]

    try:
        popt, pcov = curve_fit(double_gaussian, x, y, p0=p0, maxfev=10000)
    except RuntimeError:
        print("Double Gaussian fit did not converge.")
        return None, None

    fwhm1 = 2.3548 * abs(popt[2])
    fwhm2 = 2.3548 * abs(popt[5])

    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(x, y, 'b.', label='Data')
        xfit = np.linspace(np.min(x), np.max(x), 500)
        plt.plot(xfit, double_gaussian(xfit, *popt), 'r-', label='Double Gaussian fit')
        plt.axvline(popt[1], color='g', linestyle='--', label=f'Peak 1 center {popt[1]:.2f}')
        plt.axvline(popt[4], color='m', linestyle='--', label=f'Peak 2 center {popt[4]:.2f}')
        plt.axvspan(popt[1] - fwhm1/2, popt[1] + fwhm1/2, color='orange', alpha=0.2, label='FWHM 1')
        plt.axvspan(popt[4] - fwhm2/2, popt[4] + fwhm2/2, color='cyan', alpha=0.2, label='FWHM 2')
        plt.ylabel('Counts')
        if in_energy:
            plt.xlabel('Energy (ev)')
            plt.title(f'Double Gaussian Fit: FWHM1 = {fwhm1:.2f} ev, FWHM2 = {fwhm2:.2f} ev')
        else:
            plt.xlabel('PH values')
            plt.title(f'Double Gaussian Fit: FWHM1 = {fwhm1:.2f} PH, FWHM2 = {fwhm2:.2f} PH')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if return_sig:
        sigmas = np.sqrt(np.diag(pcov))
        return (fwhm1, fwhm2), popt, sigmas
    else :
        return (fwhm1, fwhm2), popt

def weighted_mse(errors, uncertainties, use_variance=False):
    """
    Compute the weighted MSE, where each error term is weighted by the inverse of uncertainty.
    
    Parameters:
    - errors: Array of differences (y_true - y_pred).
    - uncertainties: Array of uncertainties (standard deviations or variances) of predictions.
    - use_variance: If True, treats 'uncertainties' as variances (weights = 1/var).
                   If False, treats them as standard deviations (weights = 1/std^2).
    
    Returns:
    - Weighted MSE (scalar).
    """
    errors = np.asarray(errors)
    uncertainties = np.asarray(uncertainties)
    
    if use_variance:
        weights = 1.0 / (uncertainties + 1e-10)  # Avoid division by zero
    else:
        weights = 1.0 / (uncertainties ** 2 + 1e-10)  # Convert std to variance
    
    weighted_sq_errors = weights * (errors ** 2)
    return np.mean(weighted_sq_errors)

class Coadd_data:
    def __init__(self, data, timebase, calibration_output, ch_list, peak_e = None, peak_name = None, escp_pk_e = None):
        self.data = data
        self.ch_list = ch_list
        self.channum = "Coadded data"
        self.p_energy = np.array([])
        self.p_energy_mean = np.array([])
        self.p_energy_std = np.array([])
        self.p_filt_value_dc = np.array([])
        self.good_mask = np.array([], dtype="bool")
        # self.timestamp = np.array([])
        self.times = np.array([])
        self.p_rise_time = np.array([])
        self.p_pretrig_mean = np.array([])
        self.peak_e = peak_e
        self.escp_pk_e = escp_pk_e
        self.peak_names = peak_name

        for ch in ch_list:
            ds = data.channel[ch]
            g = ds.good()
            self.p_energy = np.append(self.p_energy, ds.p_energy[:])
            self.p_energy_mean = np.append(self.p_energy_mean, calibration_output[ch]["e_mean"])
            self.p_energy_std = np.append(self.p_energy_std, calibration_output[ch]["e_std"])
            self.p_filt_value_dc = np.append(self.p_filt_value_dc, ds.p_filt_value_dc[:])
            self.p_pretrig_mean = np.append(self.p_pretrig_mean, ds.p_pretrig_mean[:])
            self.good_mask = np.append(self.good_mask, g)
            self.p_rise_time = np.append(self.p_rise_time, ds.p_rise_time)
            # self.timestamp = np.append(self.timestamp, ds.timestamp)
            self.times = np.append(self.times, timebase[ch])
    
    def good(self):
        return self.good_mask
    
    def hist(self, energy_range=(20e3,120e3), nbins = 60000, do_plot = True, figsize = (9,4), do_cal_lines = False):
        """
        Calculates the histogram of the coadded data

        Parameters:
        - energy_range: tuple, (min, max) energy in ev
        - nbins_e: int, number of bins
        - plot: bool, whether to plot the result

        Returns:
        - ebins: bin edges (array)
        - coaddE: shape (2, nbins_e), [all events, good events]
        """
        nbins = int(nbins)

        c, b = np.histogram(self.p_energy[:], bins = nbins, range=energy_range)
        cg, bg = np.histogram(self.p_energy[self.good_mask], bins = nbins, range=energy_range)
        if do_plot:
            plt.figure(figsize=figsize)
            plt.semilogy(b[:-1], c, label='all events')
            plt.semilogy(bg[:-1], cg, label='all good')
            if do_cal_lines:
                if self.peak_names is not None and self.peak_e is not None:
                    cmap = plt.get_cmap('tab10')  # or 'tab20', 'Set1', etc.
                    for i in range(len(self.peak_e)):
                        plt.axvline(self.peak_e[i], label=self.peak_names[i], linestyle="--", color=cmap(i % cmap.N), alpha=0.6)
                else :
                    raise ValueError("No argument has been given for peak_names or peak_e")
            plt.xlim(energy_range)
            plt.xlabel('energy (ev)')
            plt.legend()
            plt.title('coadded energy spectrum')
            plt.show()
        return [c, cg], [b, bg]
    
    def peaktest(self, zbins = 600, do_plot = False):
        escp_pk_idx = []
        escp_pk_e = self.escp_pk_e
        mu_e = np.empty(len(self.peak_e))
        sigma_e = np.empty(len(self.peak_e))
        mu_e_unc = np.empty(len(self.peak_e))
        e_diff = np.empty(len(self.peak_e))
        in_range = np.empty(len(self.peak_e), dtype=bool)
        fwhm_arr = np.empty(len(self.peak_e))
        skip = False

        if escp_pk_e is not None :
            for e in escp_pk_e :
                escp_pk_idx.append(int(np.where(self.peak_e==e)[0][0]))
                # print(f"The indices of the escape peaks are {escp_pk_idx}")
        for i in range(len(self.peak_e)):
            pk_e = self.peak_e[i]
            if skip :
                skip = False
            
            elif i in escp_pk_idx:
                e_delta = 500 #in eV
                zc, zb = zoom_in_energy(self, pk_e+100, zbins=zbins, do_plot=do_plot)
                zmask = (zb > pk_e-e_delta+100) & (zb < pk_e+e_delta+100)
                zc_filt = zc[zmask[:-1]]
                zb_filt = zb[zmask]

                if do_plot:
                    plt.figure(figsize=(9,5))
                    plt.plot(zb_filt, zc_filt, "bo")
                    plt.axvline(pk_e, color="red", linestyle="--")
                    plt.title(f"Zoom on center {pk_e} ± {e_delta} eV")
                    plt.show()
            
                fwhm, popt, sigmas = fit_double_gaussian(zb_filt, zc_filt, plot=do_plot, return_sig=True, in_energy=True)
                a1, mu1, sigma1, a2, mu2, sigma2, c = popt
                a1_sig, mu1_sig, sigma1_sig, a2_sig, mu2_sig, sigma2_sig, c_sig = sigmas
                if mu1 < mu2:
                    mu_e[i] = mu1
                    mu_e[i+1] = mu2
                    sigma_e[i] = sigma1
                    sigma_e[i+1] = sigma2
                    mu_e_unc[i] = mu1_sig
                    mu_e_unc[i+1] = mu2_sig
                    fwhm_arr[i], fwhm_arr[i+1] = fwhm
                else : 
                    mu_e[i+1] = mu1
                    mu_e[i] = mu2
                    sigma_e[i+1] = sigma1
                    sigma_e[i] = sigma2
                    mu_e_unc[i+1] = mu1_sig
                    mu_e_unc[i] = mu2_sig
                    fwhm_arr[i+1], fwhm_arr[i] = fwhm

                skip = True
            
            else :
                zc, zb = zoom_in_energy(self, pk_e, zbins=zbins, do_plot=do_plot)
                e_delta = 400

                zmask = (zb > pk_e-e_delta) & (zb < pk_e+e_delta)
                zc_filt = zc[zmask[:-1]]
                zb_filt = zb[zmask]

                if do_plot:
                    plt.figure(figsize=(9,5))
                    plt.plot(zb_filt, zc_filt, "bo")
                    plt.axvline(pk_e, color="red", linestyle="--")
                    plt.title(f"Zoom on center {pk_e} ± {e_delta} eV")
                    plt.show()

                fwhm, popt, sigmas = fit_fwhm(zb_filt, zc_filt, plot=do_plot, return_sig=True, in_energy=True)
                a, mu_e[i], sigma_e[i], c = popt
                a_sig, mu_e_unc[i], sigma_sig, c_sig = sigmas
                fwhm_arr[i] = fwhm

        plt.figure(figsize=(9,5))
        for i in range(len(self.peak_e)):
            pk_e = self.peak_e[i]
            mu = mu_e[i]
            sig = sigma_e[i]
            mu_unc = mu_e_unc[i]
            fwhm = fwhm_arr[i]

            e_diff[i] = pk_e-mu
            if np.abs(e_diff[i]) < mu_unc:
                in_range[i] = True
            else :
                in_range[i] = False
            print(f"Peak {pk_e} eV, In range {in_range[i]}, Measured {mu} ± {mu_unc} eV, FWHM {fwhm} eV")
            plt.errorbar(pk_e, e_diff[i], yerr=mu_unc, fmt="o", label=f"Peak {pk_e:.2f} eV")
        plt.axhline(0, linestyle="--", color = "red")
        plt.legend()
        plt.title("Difference between known value and peak center of the coadd against energy value")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Energy difference (eV)")
        plt.show()

        print(f"The result of the MSE loss function on the errors of the peak center is {weighted_mse(e_diff, mu_e_unc, use_variance=use_variance)}")

        plt.figure(figsize=(11,7))
        for i in range(len(self.peak_e)):
            pk_e = self.peak_e[i]
            fwhm = fwhm_arr[i]
            plt.plot(pk_e, fwhm, "o", label=f"Peak {pk_e:.2f} eV")
        # plt.axhline(0, linestyle="--", color = "red")
        plt.legend()
        plt.tight_layout()
        plt.title("FWHM of each peak of the coadd")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Energy (eV)")
        plt.show()
        return mu_e, mu_e_unc, e_diff, in_range
            
    
    def plot_offset_spectra(self, energy_range=(20e3,120e3), nbins=6000, offset_base=1.3, do_log=True):
        """
        Plot all calibrated energy spectra for each channel in the Coadd_data object on the same graph with an offset.

        Parameters:
        - energy_range: tuple, x-axis limits for energy (default (1, 110))
        - nbins: int, number of bins for the histogram
        - offset_base: float, base for offset scaling (default 1.3)
        - do_log: bool, whether to use log scale for y-axis
        """
        plt.figure(figsize=(9, 6))
        start = 0
        self.all_histos = np.zeros((2, len(self.ch_list), nbins))
        for nn, ch in enumerate(self.ch_list):
            ds = self.data.channel[ch]
            g = ds.good()
            c, b = np.histogram(ds.p_energy[:], bins=nbins, range=energy_range)
            cg, bg = np.histogram(ds.p_energy[g], bins=nbins, range=energy_range)
            self.all_histos[0, nn, :] = c
            self.all_histos[1, nn, :] = cg

            offset = offset_base ** (nn*15)
            plt.plot(bg[:-1], (c + 1) * offset, label=f"Ch {ch}")

        plt.xlim(energy_range)
        plt.xlabel('energy (ev)')
        if do_log:
            plt.yscale('log')
        plt.title('Offset energy spectra by channel (Coadd_data)')
        plt.legend()
        plt.show()

class CubicSplineWithUncertainty:
    """
    A cubic spline with uncertainty bands computed via Monte Carlo.
    Usage:
    >>> spline = CubicSplineWithUncertainty(x, y, x_err, y_err, n_samples=1000)
    >>> y_mean = spline(x_eval)           # Mean spline prediction
    >>> y_std = spline.uncertainty(x_eval) # Standard deviation (1σ uncertainty)
    """
    def __init__(self, x, y, x_err=None, y_err=None, n_samples=1000):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.x_err = np.zeros_like(self.x) if x_err is None else np.asarray(x_err)
        self.y_err = np.zeros_like(self.y) if y_err is None else np.asarray(y_err)
        self.n_samples = n_samples

        # Filter out any entries where x, y, x_err, or y_err are nan
        mask = ~(
            np.isnan(self.x) |
            np.isnan(self.y) |
            np.isnan(self.x_err) |
            np.isnan(self.y_err)
        )
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.x_err = self.x_err[mask]
        self.y_err = self.y_err[mask]

        self._fit_spline_with_uncertainty()

    def _fit_spline_with_uncertainty(self):
        # Generate perturbed datasets
        x_perturbed = self.x + np.random.normal(0, self.x_err, (self.n_samples, len(self.x)))
        y_perturbed = self.y + np.random.normal(0, self.y_err, (self.n_samples, len(self.y)))

        # Fit splines to all perturbed datasets
        self.cs = CubicSpline(self.x, self.y)
        self.splines = [CubicSpline(x_perturbed[i], y_perturbed[i]) for i in range(self.n_samples)]
        self.gain_spline = CubicSpline(self.x, self.x/self.y)
        self.gain_splines = [CubicSpline(x_perturbed[i], x_perturbed[i]/y_perturbed[i]) for i in range(self.n_samples)]

    def __call__(self, x_eval):
        """Evaluate the mean spline at x_eval."""
        return self.cs(x_eval)
    
    def mean(self, x_eval):
        """Evaluate the mean spline at x_eval."""
        return np.mean([spline(x_eval) for spline in self.splines], axis=0)

    def uncertainty(self, x_eval, test=False):
        """Evaluate the standard deviation (1σ uncertainty) at x_eval."""
        if test:
            temp = np.array([spline(x_eval) for spline in self.splines])
            plt.figure(figsize=(12,7))
            plt.hist(temp, 100)
            plt.xlabel("Possible energy (ev)")
            plt.ylabel("Counts")
            plt.title("Distribution of the energies calculated at 1σ uncertainty")
            plt.show()
            c, b = np.histogram(temp, bins = 100)
            return c, b
        return np.std([spline(x_eval) for spline in self.splines], axis=0)

    def min(self, x_eval, n_sigma=1):
        self.cs_min = CubicSpline(self.x - self.x_err*n_sigma, self.y - self.y_err*n_sigma)
        return self.cs_min(x_eval)

    def max(self, x_eval, n_sigma=1):
        self.cs_max = CubicSpline(self.x + self.x_err*n_sigma, self.y + self.y_err*n_sigma)
        return self.cs_max(x_eval)
    
    def gain(self, x_eval):
        return self.gain_spline(x_eval)
    
    def g_mean(self, x_eval):
        """Evaluate the mean spline at x_eval."""
        return np.mean([spline(x_eval) for spline in self.gain_splines], axis=0)

    def g_uncertainty(self, x_eval, test=False):
        """Evaluate the standard deviation (1σ uncertainty) at x_eval."""
        if test:
            temp = np.array([spline(x_eval) for spline in self.gain_splines])
            plt.figure(figsize=(12,7))
            plt.hist(temp, 100)
            plt.xlabel("Possible gian (PH/ev)")
            plt.ylabel("Counts")
            plt.title("Distribution of the gain calculated at 1σ uncertainty")
            plt.show()
            c, b = np.histogram(temp, bins = 100)
            return c, b
        return np.std([spline(x_eval) for spline in self.gain_splines], axis=0)


def linear_fit(p, x):
    a, b = p
    return a*x + b

class LinearCalibration:
    def __init__(self, x, y, x_err=None, y_err= None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.x_err = np.zeros_like(x) if x_err is None else np.asarray(x_err)
        self.y_err = np.zeros_like(y) if y_err is None else np.asarray(y_err)
        self.fit_linear()
    
    def fit_linear(self):
        # Create a model for fitting
        linear_model = Model(linear_fit)
        data = RealData(self.x, self.y, sx=self.x_err, sy=self.y_err)
        initial_guess = self.y[2]/self.x[2]
        odr = ODR(data, linear_model, beta0=[initial_guess, 0.0])
        # Run the regression
        result = odr.run()

        # Extract fitted parameters and their standard errors
        self.a, self.b = result.beta
        self.a_err, self.b_err = result.sd_beta

        print(f"Fitted line: y = ({self.a:.3e} ± {self.a_err:.3e}) * x + ({self.b:.3f} ± {self.b_err:.3f})")

    def __call__(self, x_eval):
        p = (self.a, self.b)
        return linear_fit(p, x_eval)
    
    def return_parameter(self):
        return self.a, self.b, self.a_err, self.b_err



def horizontal_distances_to_neighbors(points_x, idx):
    """
    Calculate the horizontal distances from the removed calibration point
    to its two neighboring calibration points.

    Parameters:
    - points_x: array-like, PH values of calibration points (must be 1D)
    - idx: int, index of the calibration point

    Returns:
    - left_distance: distance to the left neighbor (0 if no left neighbor)
    - right_distance: distance to the right neighbor (0 if no right neighbor)
    """
    n = len(points_x)
    left_distance = 0
    right_distance = 0

    if idx > 0:
        left_distance = abs(points_x[idx] - points_x[idx - 1])
    if idx < n - 1:
        right_distance = abs(points_x[idx] - points_x[idx + 1])

    return left_distance, right_distance



def data_treat(run_pulse, run_noise, files_to_delete):
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")

    pulsePattern = sorted(glob.glob(run_pulse))
    noisePattern = sorted(glob.glob(run_noise))
    data = mass.TESGroup(pulsePattern, noisePattern) #load the data into mass 

    #apply base MASS analysis algorithms 
    data.compute_noise(forceNew=True)
    data.summarize_data( forceNew=True)
    data.calc_external_trigger_timing(forceNew=True)

    for ds in data: ds.clear_cuts()
    data.auto_cuts()
    data.correct_flux_jumps(4096)
    data.avg_pulses_auto_masks(forceNew=True)

    data.compute_5lag_filter(f_3db=20000,forceNew=True)
    data.filter_data(forceNew=True)
    data.drift_correct(forceNew=True)
    data.phase_correct(forceNew=True)

    # After data.auto_cuts()
    print("Good channels:", data.good_channels)
    print("All channels:", list(data.channel.keys()))
    timebase = {}
    for ch in data.channel:
        ds = data.channel[ch]
        timebase[ch] = np.array(ds.subframes_after_last_external_trigger) * data.subframe_timebase * 1000
    return data, timebase

# Finding the peaks in the zoomed histogram
def log_find_peaks(counts, height=None, distance=None, prominence=None):
    """
    Finds peaks in the zoomed histogram data.
    
    Parameters:
    - counts: Histogram counts
    - bin_edges: Bin edges of the histogram
    - height: Minimum height of peaks
    - distance: Minimum distance between peaks
    - prominence: Minimum prominence of peaks
    
    Returns:
    - peaks: Indices of the peaks found
    - properties: Properties of the peaks found
    """
    log_counts = np.log10(counts + 1)
    peaks, properties = find_peaks(log_counts, height=height, distance=distance, prominence=prominence)
    
    return peaks, properties


# Adds the histogram of a channel to an existing matplotlib figure/axes.
def add_channel_histogram_to_ax(data, ch, ax, bins=5000, hist_range=(0, 4e3), Timebase = None, do_plot = True, do_peaks = True, do_allg=False, return_val=False, height=None, distance=None, prominence=None):
    """
    Add the histogram of a channel to an existing matplotlib Axes.
    Can return the counts, the bins, the peaks' indixes and their properties if return_val = True

    Parameters:
    - data: MASS TESGroup object
    - ch: channel number (int)
    - ax: matplotlib Axes object
    - bins: number of bins for the histogram
    - hist_range: tuple, range for the histogram
    """
    ds = data.channel[ch]
    g = ds.good()
    
    c, b = np.histogram(ds.p_filt_value_dc[:], bins=bins, range=hist_range)
    cg, bg = np.histogram(ds.p_filt_value_dc[g], bins=bins, range=hist_range)
    ctg, btg, peaks, properties = None, None, None, None

    if do_plot :
        # Plot on the provided axes
        if do_allg :
            ax.plot(b[:-1], c, label="all")
            ax.plot(bg[:-1], cg, label="good")
        if Timebase is not None :
            mask = Timebase[ch] > 100
            gmask =  mask & g
            ctg, btg, = np.histogram(ds.p_filt_value_dc[gmask], bins=bins, range=hist_range)
            ax.plot(btg[:-1], ctg, label="good and trigger filtered")
        if do_peaks :
            # Find peaks for annotation (optional)
            peaks, properties = log_find_peaks(ctg, height=height, distance=distance, prominence=prominence)
            hauteurs = ctg[peaks]
            positions = btg[:-1][peaks]
            ax.plot(positions, hauteurs, "rx", label="peaks")
        ax.xlabel("PH values")
        ax.ylabel("Comptes")
        ax.title(f'ch {ds.channum}')
        ax.xlim((0, 4e3))
        ax.yscale("log")
        ax.legend()

    if return_val == True:
        return ctg, btg, peaks, properties
    
# --- Find escape peaks ---
def find_biggest_peak(counts, bin_edges):
    peaks, properties = find_peaks(counts, height=0)
    if len(peaks) == 0:
        return None, None, None
    biggest_peak_idx = np.argmax(properties['peak_heights'])
    peak_index = peaks[biggest_peak_idx]
    peak_ph = 0.5 * (bin_edges[peak_index] + bin_edges[peak_index + 1])
    peak_count = counts[peak_index]
    return peak_index, peak_ph, peak_count

def get_bin_count(value, counts, bin_edges):
    bin_index = np.searchsorted(bin_edges, value, side='right') - 1
    if bin_index < 0 or bin_index >= len(counts):
        return 0
    return counts[bin_index]

def zoom_in_ph(ds, ph_value, zoom_factor=0.1, zbins=200, do_plot = True):
    """
    Zoom in on a given PH value and show the zoomed histogram and the full spectrum.

    Parameters:
    - ds: MASS dataset for a channel
    - ph_value: PH value to center the zoom on
    - zoom_factor: relative width of zoom window (fraction of 1000)
    - zbins: number of bins for zoomed histogram
    - hheight: whether to plot the half-height line
    """
    g = ds.good()
    values = ds.p_filt_value_dc[g]
    ch = ds.channum

    window = 1000
    zoom_min = ph_value - window * zoom_factor
    zoom_max = ph_value + window * zoom_factor
    mask = (values >= zoom_min) & (values <= zoom_max)
    zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))

    if do_plot:
        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Full spectrum with the selected PH value highlighted
        full_counts, full_bins = np.histogram(values, bins=6000, range=(0, 4e3))
        axs[0].plot(full_bins[:-1], full_counts, label="Full good spectrum")
        axs[0].axvline(ph_value, color='red', linestyle='--', label='Selected PH')
        axs[0].set_xlabel("PH Value")
        axs[0].set_ylabel("Counts")
        axs[0].set_title(f"Full spectrum (ch {ch})")
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[0].set_xlim(full_bins[0], full_bins[-1])

        # Right: Zoomed-in region
        axs[1].plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
        axs[1].axvline(ph_value, color='red', linestyle='--', label='Selected PH', alpha=0.3)
        axs[1].set_xlabel("PH Value")
        axs[1].set_ylabel("Counts")
        axs[1].set_title(f"Zoom on PH {ph_value:.1f}")
        axs[1].set_xlim(zoom_min, zoom_max)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    return zoom_counts, zoom_bin_edges

def zoom_in_energy(ds, energy_value, zoom_factor=0.1, zbins=200, do_plot= True):
    """
    Zoom in on a given Energy value (in eV) and show the zoomed histogram and the full energy spectrum.

    Parameters:
    - ds: MASS dataset for a channel
    - energy_value: Energy value (ev) to center the zoom on
    - zoom_factor: relative width of zoom window (fraction of 20e3)
    - zbins: number of bins for zoomed histogram
    - hheight: whether to plot the half-height line
    """
    g = ds.good()
    values = ds.p_energy[g]
    ch = ds.channum

    window = 20e3
    zoom_min = energy_value - window * zoom_factor
    zoom_max = energy_value + window * zoom_factor
    mask = (values >= zoom_min) & (values <= zoom_max)
    zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))

    if do_plot:
        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Full energy spectrum with the selected energy highlighted
        full_counts, full_bins = np.histogram(values, bins=6000, range=(0e3, 150e3))
        axs[0].plot(full_bins[:-1], full_counts, label="Full good spectrum")
        axs[0].axvline(energy_value, color='red', linestyle='--', label='Selected Energy')
        axs[0].set_xlabel("Energy (ev)")
        axs[0].set_ylabel("Counts")
        axs[0].set_title(f"Full energy spectrum (ch {ch})")
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[0].set_xlim(full_bins[0], full_bins[-1])

        # Right: Zoomed-in region
        axs[1].plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
        axs[1].axvline(energy_value, color='red', linestyle='--', label='Selected Energy', alpha=0.3)
        axs[1].set_xlabel("Energy (ev)")
        axs[1].set_ylabel("Counts")
        axs[1].set_title(f"Zoom on Energy {energy_value:.1f} ev")
        axs[1].set_xlim(zoom_min, zoom_max)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    return zoom_counts, zoom_bin_edges


def zoom_in_on_peaks(ds, peak_data, peak_index, zoom_factor=0.1, zbins=200, hheight=True, do_plot = True):
    """
    Display the zoomed-in peak and, next to it, the full spectrum with the selected peak highlighted.

    Parameters:
    - ds: MASS dataset for a channel
    - peak_data: list of [calp_dict, count_dict, ph_val_dict]
    - peak_index: which peak to zoom in on
    - zoom_factor: relative width of zoom window
    - zbins: number of bins for zoomed histogram
    - hheight: whether to plot the half-height line
    """

    g = ds.good()
    values = ds.p_filt_value_dc[g]
    ch = ds.channum

    peak_idx_list = peak_data[0][ch]
    count_list = peak_data[1][ch]
    ph_list = peak_data[2][ch]

    chosen_peak_idx = peak_index
    chosen_peak_count = ph_list[chosen_peak_idx]
    chosen_peak_height = count_list[chosen_peak_idx]

    window = 1000
    zoom_min = chosen_peak_count - window * zoom_factor
    zoom_max = chosen_peak_count + window * zoom_factor
    mask = (values >= zoom_min) & (values <= zoom_max)
    zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))
    zoom_peak_height = get_bin_count(chosen_peak_count, zoom_counts, zoom_bin_edges)
    half_height = zoom_peak_height / 2

    if do_plot :
        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Full spectrum with all peaks, highlight selected
        full_counts, full_bins = np.histogram(values, bins=6000, range=(0, 4e3))
        axs[0].plot(full_bins[:-1], full_counts, label="Full good spectrum")
        axs[0].plot(chosen_peak_count, chosen_peak_height, 'rx', label='Selected peak')
        axs[0].set_xlabel("PH Value")
        axs[0].set_ylabel("Counts")
        axs[0].set_title(f"Full spectrum (ch {ch})")
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[0].set_xlim(full_bins[0], full_bins[-1])

        # Right: Zoomed-in peak
        axs[1].plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
        axs[1].axvline(chosen_peak_count, color='red', linestyle='--', label='Selected peak', alpha=0.3)
        if hheight:
            axs[1].axhline(half_height, color='green', linestyle='--', label='Half height', alpha=0.3)
        axs[1].set_xlabel("PH Value")
        axs[1].set_ylabel("Counts")
        axs[1].set_title(f"Zoom on peak {peak_index} at {chosen_peak_count:.1f}")
        axs[1].set_ylim(0, 1.2 * zoom_peak_height)
        axs[1].set_xlim(zoom_min, zoom_max)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    return zoom_counts, zoom_bin_edges

def zoom_all_e(data, ch, timebase=None, zoom_factor=0.1, zbins=200, hheight=True, return_val=False, distance = 30, prominence = 1, height = None):
    """
    Identifies the peaks, converts the PH into energy (eV) with an approximate linear calibration and zooms on each peak
    Parameters:
    - data : TESGroup data object
    - ch : int, channel number
    - timebase : array-like, time data
    - zoom_factor: relative width of zoom window
    - zbins: number of bins for zoomed histogram
    - hheight: whether to plot the half-height line
    """
    ds = data.channel[ch]
    g = ds.good()
    ch = ds.channum
    plt.figure(figsize=(9,5))
    ctg, btg, peaks, _ = add_channel_histogram_to_ax(data, ch, plt, return_val=True, Timebase=timebase, do_allg=False, do_peaks=True, distance=distance, prominence=prominence, height = height)
    plt.show()
    ph_list = btg[peaks]

    zoom_counts_list = []
    zoom_bin_edges_list = []
    # Simple linear calibration
    main_val = btg[np.argmax(ctg)] # ADC values
    coeff = main_e/main_val # ev/PH
    energy_list = ph_list*coeff
    values = ds.p_filt_value_dc[g]*coeff
    window = 20e3

    print(f"The coefficient is {coeff}")
    print(f"The PH of the calibration peaks is {ph_list}")
    # Zoom on every peaks
    for e in energy_list:
        zoom_min = e - window * zoom_factor
        zoom_max = e + window * zoom_factor
        mask = (values >= zoom_min) & (values <= zoom_max)
        zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))

        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Full energy spectrum with the selected energy highlighted
        full_counts, full_bins = np.histogram(values, bins=6000, range=(0, 110e3))
        axs[0].plot(full_bins[:-1], full_counts, label="Full good spectrum")
        axs[0].axvline(e, color='red', linestyle='--', label='Selected Energy')
        axs[0].set_xlabel("Energy (ev)")
        axs[0].set_ylabel("Counts")
        axs[0].set_title(f"Full energy spectrum (ch {ch})")
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[0].set_xlim(full_bins[0], full_bins[-1])

        # Right: Zoomed-in region
        axs[1].plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
        axs[1].axvline(e, color='red', linestyle='--', label='Selected Energy', alpha=0.3)
        if hheight:
            hc = int(get_bin_count(e, zoom_counts, zoom_bin_edges)/2)
            axs[1].axhline(hc, color='green', linestyle='--', label="Half height", alpha = 0.3)
        axs[1].set_xlabel("Energy (ev)")
        axs[1].set_ylabel("Counts")
        axs[1].set_title(f"Zoom on Energy {e:.1f} ev")
        axs[1].set_xlim(zoom_min, zoom_max)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        zoom_counts_list.append(zoom_counts)
        zoom_bin_edges_list.append(zoom_bin_edges)

    if return_val:
        return zoom_counts_list, zoom_bin_edges_list



class Calibration :
    def __init__(self, data, peak_e, pk_name, e_unc=None, Timebase=None, escp_pk_e= None):
        self.data= data
        self.peak_e = peak_e
        self.e_unc = e_unc
        self.timebase = Timebase
        self.escp_pk_e = escp_pk_e
        self.pk_name = pk_name
        
    def __call__(self, ch, zoom_bins = 700, do_plot = True, cal_type = "MASS"):
        # Treating the inputed data
        ds = self.data.channel[ch]
        peak_e = self.peak_e
        escp_pk_e = self.escp_pk_e
        Timebase = self.timebase
        pk_name = self.pk_name
        g = ds.good()
        ph = ds.p_filt_value_dc[:]
        zbins = zoom_bins
        e_unc = self.e_unc

        if Timebase is not None:
            mask = Timebase[ch] > 100
            gmask =  mask & g
        else : gmask = g
        ph_filtered = ph[gmask]

        escp_pk_idx = []
        if escp_pk_e is not None :
            for e in escp_pk_e :
                escp_pk_idx.append(int(np.where(peak_e==e)[0][0]))
            print(f"The indices of the escape peaks are {escp_pk_idx}")

        allowed_cal_type = ["MASS", "CubicSpline"]
        if cal_type == "MASS":
            use_mass = True
        elif cal_type == "CubicSpline":
            use_mass = False
        else :
            raise ValueError(f"Expected a value of {allowed_cal_type} but got {cal_type} for parameter cal_type")
        

        figsize = (9,6)
        mpeak_e = main_e # eV, Initializing the value for the main peak of 241am

        # Making a very discrete spectrum of the inputed data
        nbins = int(15e3)
        bin_range= (0,4e3)
        c, b = np.histogram(ph, bins =nbins, range = bin_range)
        cg, bg = np.histogram(ph_filtered, bins= nbins, range = bin_range)

        if do_plot:
            # Visualizing the spectrum
            plt.figure(figsize=figsize)
            plt.plot(b[:-1], c, label="all")
            plt.plot(bg[:-1],cg, label="filtered")
            plt.xlabel("PH values")
            plt.ylabel("Comptes")
            plt.title(f'ch {ch}')
            plt.xlim(bin_range)
            plt.yscale("log")
            plt.legend()

        # Showing the ph value for the main peak
        mpeak_ph = bg[np.argmax(cg)]
        print(f"For the main peak ({mpeak_e} eV) we have a first approximative PH value of {mpeak_ph} PH \n")

        # Converting the list of given energy values to PH
        ev_to_ph = mpeak_ph/mpeak_e
        print(f"The linear coefficient are :\n - {ev_to_ph} PH/eV \n - {1/ev_to_ph} eV/PH\n")
        peak_ph = peak_e*ev_to_ph

        # Finding the precise PH values for each peak
        mu_ph = np.empty(len(peak_ph))
        mu_ph_sig = np.empty(len(peak_ph))
        sigma_ph = np.empty(len(peak_ph))
        skip = False
        for i in range(len(mu_ph)):
            if skip :
                skip = False

            elif i in escp_pk_idx:
                zc, zb = zoom_in_ph(ds, peak_ph[i]+5, zbins=zbins, do_plot=do_plot)
                ph_delta = 15
                max_count = 1
                zoom_tries = 0
                do_fit = True
                while max_count<= 2:
                    print(f"Zoom try n°{zoom_tries}, ")
                    zmask = (zb > peak_ph[i]-ph_delta+5) & (zb < peak_ph[i]+ph_delta+5)
                    zc_filt = zc[zmask[:-1]]
                    zb_filt = zb[zmask]
                    max_count = max(zc_filt)
                    ph_delta += 10
                    if zoom_tries > 4:
                        # print(f"Couldn't find peak near {peak_ph[i]} PH")
                        # do_fit = False
                        # break
                        raise StopIteration(f"Couldn't find peak near {peak_ph[i]} PH")
                    zoom_tries += 1
                    if zoom_tries> 1:
                        if do_plot:
                            plt.figure(figsize=figsize)
                            plt.plot(zb_filt, zc_filt, "bo")
                            plt.axvline(peak_ph[i], color="red", linestyle="--")
                            plt.title(f"Zoom on center {peak_ph[i]} ± {ph_delta} PH")
                            plt.show()
                if do_fit:
                    fwhm, popt, sigmas = fit_double_gaussian(zb_filt, zc_filt, plot=do_plot, return_sig=True)
                    a1, mu1, sigma1, a2, mu2, sigma2, c = popt
                    a1_sig, mu1_sig, sigma1_sig, a2_sig, mu2_sig, sigma2_sig, c_sig = sigmas
                if mu1 < mu2:
                    mu_ph[i] = mu1
                    mu_ph[i+1] = mu2
                    sigma_ph[i] = sigma1
                    sigma_ph[i+1] = sigma2
                    mu_ph_sig[i] = mu1_sig
                    mu_ph_sig[i+1] = mu2_sig
                else : 
                    mu_ph[i+1] = mu1
                    mu_ph[i] = mu2
                    sigma_ph[i+1] = sigma1
                    sigma_ph[i] = sigma2
                    mu_ph_sig[i+1] = mu1_sig
                    mu_ph_sig[i] = mu2_sig
                if mu_ph_sig[i] > 0.7 or mu_ph_sig[i+1] > 0.7:
                    raise ValueError(f"The uncertainties on the double peak's center are too high (peaks {i} and {i+1} with uncertainties {mu_ph_sig[i]:.2f} and {mu_ph_sig[i+1]:.2f} PH), failed to calibrate")  
                skip = True
            
            else :
                if peak_ph[i] > 2000:
                    zbins = 400
                else :
                    zbins = zoom_bins
                zc, zb = zoom_in_ph(ds, peak_ph[i], zbins=zbins, do_plot=do_plot)
                ph_delta = 15
                max_count = 1
                zoom_tries = 0
                do_fit = True
                while max_count<= 2:
                    print(f"Zoom try n°{zoom_tries}, ")
                    zmask = (zb > peak_ph[i]-ph_delta) & (zb < peak_ph[i]+ph_delta)
                    zc_filt = zc[zmask[:-1]]
                    zb_filt = zb[zmask]
                    max_count = max(zc_filt)
                    ph_delta += 10
                    if zoom_tries > 4:
                        # print(f"Couldn't find peak near {peak_ph[i]} PH")
                        # do_fit = False
                        # break
                        raise StopIteration(f"Couldn't find peak near {peak_ph[i]} PH")
                    zoom_tries += 1
                    if zoom_tries> 1:
                        if do_plot:
                            plt.figure(figsize=figsize)
                            plt.plot(zb_filt, zc_filt, "bo")
                            plt.axvline(peak_ph[i], color="red", linestyle="--")
                            plt.title(f"Zoom on center {peak_ph[i]} ± {ph_delta} PH")
                            plt.show()
                if do_fit:
                    fwhm, popt, sigmas = fit_fwhm(zb_filt, zc_filt, plot=do_plot, return_sig=True)
                    a, mu_ph[i], sigma_ph[i], c = popt
                    a_sig, mu_ph_sig[i], sigma_sig, c_sig = sigmas
                    if mu_ph_sig[i] > 0.7:
                        raise ValueError(f"The uncertainties on the peak's center are too high (peak {i} with uncertainty of {mu_ph_sig[i]:.2f} PH), failed to calibrate")  
                else :
                    mu_ph[i], sigma_ph[i], mu_ph_sig[i] = None, None, None

        print(f"The PH values of the peaks are : {mu_ph}")
        print(f"The uncertainties for the peak's PH values are :{mu_ph_sig}")
        if np.any(mu_ph_sig > 0.7):
            raise ValueError("The uncertainties on the peak's center are too high, failed to calibrate")    

        y_unc = None
        if mu_ph_sig is not None or e_unc is not None:
            approximate = True
        else : 
            approximate = False
        if use_mass:
            mass_cal_maker = mass.EnergyCalibrationMaker(mu_ph, peak_e, mu_ph_sig, e_unc, pk_name)
            mass_cal = mass_cal_maker.make_calibration(curvename="linear", approximate = approximate)
            x_cal = np.linspace(mu_ph[0]-100, mu_ph[-1]+100, 1000)
            y_cal = mass_cal(x_cal)                 # The energy value according to the mass calibration
            if approximate :
               y_unc = mass_cal.ph2uncertainty(x_cal)      # The uncertainties on the energy values

        # Calibrating the Calibration curve with a cubic spline object
        if not use_mass:
            cs = CubicSplineWithUncertainty(mu_ph, peak_e, x_err=mu_ph_sig, y_err=e_unc)

            x_cal = np.linspace(mu_ph[0]-100, mu_ph[-1]+100, 1000)
            if approximate:
                y_cal = cs.mean(x_cal)        # The mean energy value for the cubicspline with incertitudes
                y_unc = cs.uncertainty(x_cal)      # The std energy value for the cubicspline with incertitudes
            else : 
                y_cal = cs(x_cal)                 # The energy value according to a simple cubicspline
            
            

        


        # # Visualizing the difference between the theorical value and the predicted energy
        # upper_range = cs.mean(mu_ph)+cs.uncertainty(mu_ph)
        # lower_range = cs.mean(mu_ph)-cs.uncertainty(mu_ph)
        # if do_plot:
        #     plt.figure(figsize=figsize)
        #     plt.plot(mu_ph, peak_e-cs(mu_ph), "bo-", label="Theorical - gain")
        #     plt.plot(mu_ph, peak_e-cs.mean(mu_ph), "ro-", label="Theorical - mean")
        #     plt.fill_between(mu_ph,peak_e-lower_range, peak_e-upper_range, alpha=0.3, label='Uncertainty (1σ)')
        #     plt.xlabel("PH Value")
        #     plt.ylabel("Theorical - Predicted Energy (ev)")
        #     plt.title("Difference between the theorical and the predicted energy")
        #     plt.legend()
        #     plt.show()

        # print("The differences between the theorical values and the ones predicted by the mean cubic spline are :")
        # for i in range(len(peak_e)):
        #     print(f"The difference for the point {i} is : {peak_e[i]-cs.mean(mu_ph[i])} ev")


        # Visualizing the difference between the predicted
        # mean_delta = y_mean-y_cal
        # stdmin_delta = y_mean-y_std-y_cal
        # stdmax_delta = y_mean+y_std-y_cal

        # if do_plot:
        #     plt.figure(figsize=figsize)
        #     plt.plot(x_cal, mean_delta, label="Mean spline - Calibration spline")
        #     plt.plot(x_cal, stdmin_delta, label="Uncertainty min - Calibration spline")
        #     plt.plot(x_cal, stdmax_delta, label="Uncertainty max - Calibration spline")
        #     plt.axhline(0, linestyle="--", color ="red",  alpha=0.3)
        #     plt.title("Difference between the different predicted values")
        #     plt.xlabel("PH values")
        #     plt.ylabel("Energy delta (ev)")
        #     plt.xlim(mu_ph[0], mu_ph[-1])
        #     plt.ylim(-0.025, 0.025)
        #     plt.legend()
        #     plt.show()

        # Visualizing the ph value against the uncertainties
        if do_plot:
            plt.figure(figsize=(13,9))
            plt.plot(mu_ph, mu_ph_sig, "bo")
            # plt.legend()
            plt.xlabel("PH Value")
            plt.ylabel("Delta PH")
            plt.title("PH value vs uncertainty")
            plt.show()

        
        # # Filtering the peaks with too large of an uncertainty
        # # e_sig = cs.uncertainty(mu_ph)
        # e_sig = mass_cal.ph2uncertainty(mu_ph)
        # mask_sig = e_sig < 2 # 2 eV
        # mu_ph_filt = mu_ph[mask_sig]
        # peak_e_filt = peak_e[mask_sig]
        # mu_ph_sig_filt = mu_ph_sig[mask_sig]
        # e_unc_filt = e_unc[mask_sig]

        # cs_filt = CubicSplineWithUncertainty(mu_ph_filt, peak_e_filt, x_err=mu_ph_sig_filt, y_err=e_unc_filt)
        
        # y_cal_filt = cs_filt(x_cal)
        # y_mean_filt = cs_filt.mean(x_cal)
        # y_std_filt = cs_filt.uncertainty(x_cal)

        # # Visualizing the predicted value and mean against the uncertainties
        # if not filter_uncertainty:
        #     if do_plot:
        #         plt.figure(figsize=figsize)
        #         plt.plot(cs(mu_ph), e_sig, "bo", label="Cubic spline")
        #         plt.plot(cs.mean(mu_ph), e_sig, "ro", label="Cubic spline mean")
        #         plt.legend()
        #         plt.xlabel("Energy (ev)")
        #         plt.ylabel("Delta E (ev)")
        #         plt.title("Predicted value and mean vs uncertainty")
        #         plt.show()

        # Calibrating on the filtered peaks
        # else :    
        
        if use_mass:
            mass_cal.plot()
        else :
            # if do_plot:
            # Visualizing the Calibration curve
            plt.figure(figsize=figsize)
            plt.errorbar(mu_ph, peak_e,xerr=mu_ph_sig, yerr=e_unc, fmt="rx")
            plt.plot(x_cal, y_cal, label = "Calibration spline")
            # plt.plot(x_cal, y_mean, "y--", label="Mean spline")
            if approximate:
                plt.fill_between(x_cal, y_cal - y_unc, y_cal + y_unc, alpha=0.3, label='Uncertainty (1σ)')
            plt.legend()
            plt.xlabel('PH Values')
            plt.ylabel('Energy values (ev)')
            plt.title(f'Calibration curve of Channel {ch}')
            plt.show()

        # Calculating the energy of the spectrum
        # if not filter_uncertainty:
        #     mask_ph = (ds.p_filt_value_dc[:] > mu_ph[0] - 100) & (ds.p_filt_value_dc[:] < mu_ph[-1] + 100)
        #     mask_ph = ~mask_ph

        #     print(f"The first PH value considered is {mu_ph[0]:.2f} and the last one is {mu_ph[-1]:.2f}")

        #     if np.any(mask_ph):
        #         print("At least one element is selected by the mask.")
        #     else:   
        #         print("No elements are selected by the mask.")  
        #     print("Number of elements selected:", np.sum(mask_ph))

        #     ds.p_energy[:] = cs(ds.p_filt_value_dc[:])
        #     e_mean = cs.mean(ds.p_filt_value_dc[:])
        #     e_std = cs.uncertainty(ds.p_filt_value_dc[:])
        #     ds.p_energy[mask_ph] = None
        #     e_mean[mask_ph] = None
        #     e_std[mask_ph] = None

        #     is_all_none = np.all(ds.p_energy[:]==None)
        #     is_all_nan = np.all(np.isnan(ds.p_energy[:]))
        #     print(f"The energy array is full of None values : {is_all_none}")
        #     print(f"The energy array is full of NaN values : {is_all_nan}")
        #     print(f"The energy array has nan values : {np.isnan(ds.p_energy[:]).any()}")
        #     if is_all_none or is_all_nan:
        #         raise ValueError("The channel doesn't have any energy values")
        #     print(f"\n The channel is calibrated between {min(ds.p_energy[:]):.3f} and {max(ds.p_energy[:]):.3f} ev")

        #     # Plotting
        #     c_e, b_e = np.histogram(ds.p_energy[:], bins=6000, range=(np.nanmin(ds.p_energy[:]), np.nanmax(ds.p_energy[:])))
        #     cg_e, bg_e = np.histogram(ds.p_energy[g], bins=6000, range=(np.nanmin(ds.p_energy[:]), np.nanmax(ds.p_energy[:])))
        #     plt.figure(figsize=figsize)
        #     plt.plot(b_e[:-1], c_e, label="all")
        #     plt.plot(bg_e[:-1], cg_e, label="good")
        #     plt.xlabel("Energy (ev)")
        #     plt.ylabel("Counts")
        #     plt.title(f'Channel {ch} Energy Spectrum (Cubic Spline Calibration)')
        #     plt.legend()
        #     plt.yscale("log")
        #     plt.show()

        # else :
        mask_ph = (ds.p_filt_value_dc[:] > mu_ph[0] - 100) & (ds.p_filt_value_dc[:] < mu_ph[-1] + 100)
        mask_ph = ~mask_ph
        print(f"The first PH value considered is {mu_ph[0]:.2f} and the last one is {mu_ph[-1]:.2f}")

        if np.any(mask_ph):
            print("At least one element is selected by the mask.")
        else:   
            raise ValueError(f"No elements are selected by the mask.") 
        print("Number of elements selected:", np.sum(mask_ph))

        e_mean, e_std = None, None
        if use_mass:
            ds.p_energy[:] = mass_cal(ds.p_filt_value_dc[:])
            if approximate:
                e_mean = mass_cal(ds.p_filt_value_dc[:])
                e_std = mass_cal.ph2uncertainty(ds.p_filt_value_dc[:])
        else :
            ds.p_energy[:] = cs(ds.p_filt_value_dc[:])
            if approximate:
                e_mean = cs.mean(ds.p_filt_value_dc[:])
                e_std = cs.uncertainty(ds.p_filt_value_dc[:])

        ds.p_energy[mask_ph] = None
        e_mean[mask_ph] = None
        e_std[mask_ph] = None

        is_all_none = np.all(ds.p_energy[:]==None)
        is_all_nan = np.all(np.isnan(ds.p_energy[:]))
        print(f"The energy array is full of None values : {is_all_none}")
        print(f"The energy array is full of NaN values : {is_all_nan}")
        print(f"The energy array has nan values : {np.isnan(ds.p_energy[:]).any()}")
        if is_all_none or is_all_nan:
            raise ValueError("The channel doesn't have any energy values")
        print(f"\n The channel is calibrated between {min(ds.p_energy[:]):.3f} and {max(ds.p_energy[:]):.3f} ev")

        c_e, b_e = np.histogram(ds.p_energy[:], bins=6000, range=(np.nanmin(ds.p_energy[:]), np.nanmax(ds.p_energy[:])))
        cg_e, bg_e = np.histogram(ds.p_energy[g], bins=6000, range=(np.nanmin(ds.p_energy[:]), np.nanmax(ds.p_energy[:])))
        plt.figure(figsize=figsize)
        plt.plot(b_e[:-1], c_e, label="all")
        plt.plot(bg_e[:-1], cg_e, label="good")
        plt.xlabel("Energy (ev)")
        plt.ylabel("Counts")
        plt.title(f'Channel {ch} Energy Spectrum (Cubic Spline Calibration)')
        plt.legend()
        plt.yscale("log")
        plt.show()
        

        if (np.nanmin(ds.p_energy[:]) < peak_e[0]-10e3) or (np.nanmax(ds.p_energy[:]) > peak_e[-1] +10e3 ):
            raise ValueError("The calibration went out of bounds")
        if use_mass:
            self.output = {
                "mass_cal" : mass_cal,
                # "cs" : cs,
                # "cs_filt" : cs_filt,
                "mu_ph" : mu_ph,
                "mu_ph_sig" : mu_ph_sig,
                "e_mean" : e_mean,
                "e_std" : e_std,
                # "mask_sig" : mask_sig,
                "hist_e" : [cg_e, bg_e],
            }
        else :
            self.output = {
                # "mass_cal" : mass_cal,
                "cs" : cs,
                # "cs_filt" : cs_filt,
                "mu_ph" : mu_ph,
                "mu_ph_sig" : mu_ph_sig,
                "e_mean" : e_mean,
                "e_std" : e_std,
                # "mask_sig" : mask_sig,
                "hist_e" : [cg_e, bg_e],
            }
        return self.output

def Calibrate_all(data, peak_e, e_unc=None, Timebase=None, escp_pk_e= None, do_plot=False, pk_name = None, cal_type = "MASS"):
    """
    Function that calibrates all of the data with chosen calibration points

    Parameters :
    - data : The mass TESGROUP data object
    - peak_e : An array of the calibration peak's energy in eV
    - e_unc : An array containing the uncertainties of the calibration peak's energy in eV
    - Timebase : An array containing all time related info of the data
    - escp_pk_e : An array containing the energies of escape peaks used for the calibration
    - do_plot : A boolean to choose to plot the graphs of the calibration
    - filter_uncertainty : A boolean to choose to filter out calibration points with a final uncertainty above 2 eV (not recommended)
    - cal_type : A string, either "MASS" or "CubicSpline", chooses which method to use for the computation of the splines

    Returns :
    cal : The calibration object
    coadded_data : The Coadd_data object which has almost the same properties as an individual channel object (works in the same way for all the functions in this library)
    outputs : A list of the outputs dictionnary resulting from the calibration of each channel
    calibrated_ch : A list of all the channels that were calibrated succesfully
    failed_ch : A list of all the channels that were calibrated unsuccesfully
    error_log : A list of all the reasons the unsuccesfull channels weren't calibrated
    """
    #Initializing the calibration
    cal = Calibration(data, peak_e, e_unc=e_unc, Timebase=Timebase, escp_pk_e=escp_pk_e, pk_name=pk_name)
    outputs = {}
    calibrated_ch= []
    failed_ch = []
    error_log = {}
    for ch in data.good_channels:
        try:
            print(f"\n\n\nCalibration of channel {ch}\n\n\n")
            outputs[ch] = cal(ch, do_plot=do_plot, cal_type=cal_type)
            calibrated_ch.append(ch)
        except Exception as e:
            failed_ch.append(ch)
            error_log[ch] = e
            print(f"Failed to calibrate channel {ch} due to : {e} \n Passing onto the next one")
    coadded_data = Coadd_data(data, Timebase, outputs, calibrated_ch, peak_e=peak_e, peak_name=pk_name, escp_pk_e=escp_pk_e)
    return cal, coadded_data, outputs, calibrated_ch, failed_ch, error_log



def loo(
    calib_output, peak_e, e_unc = None, do_plot=True, verbose=True, use_mask=False
):
    """
    Leave-one-out study for CubicSplineWithUncertainty.
    For each calibration point that was used in the original fit, fit a spline leaving it out,
    predict its energy, and check if the true value is within the 1σ error band.
    Optionally plots the results.

    Parameters:
    - calib_output: dict, output from calibration()
    - peak_e: array-like, true energy values for each peak
    - do_plot: bool, plot the results
    - verbose: bool, print details
    - used_mask: array-like of bool, same length as peak_e. True for points used in the original spline.
    """
    calib_output = calib_output
    peak_e = peak_e
    mu_ph = np.array(calib_output["mu_ph"])
    mu_ph_sig = np.array(calib_output["mu_ph_sig"])
    n = len(mu_ph)

    if use_mask:
        used_mask = np.array(calib_output["mask_sig"])
    else :
        used_mask = np.ones(n, dtype=bool)

    pred_list = []
    pred_mean_list = []
    pred_std_list = []
    within_list = []
    within2_list = []
    error_std_list = []
    loo_indices = np.where(used_mask)[0]
    e_unc = e_unc

    for i in range(n):
        mask = used_mask.copy()
        if i in loo_indices:
            mask[i] = False  # leave this one out
            cs_loo = CubicSplineWithUncertainty(mu_ph[mask], peak_e[mask], x_err=mu_ph_sig[mask], y_err=e_unc[mask])
            pred = cs_loo(mu_ph[i])
            pred_mean = cs_loo.mean(mu_ph[i])
            pred_std = cs_loo.uncertainty(mu_ph[i])
            theor = peak_e[i]
            within = abs(theor - pred_mean) <= pred_std
            within2 = abs(theor - pred_mean) <= 2*pred_std
            error_std = theor - pred_mean
            if verbose:
                print(f"Removed peak {i}: PH={mu_ph[i]:.2f}, True={theor:.4f} ev, "
                    f"Predicted ={pred_mean:.4f} ± {pred_std:.4f} ev, "
                    f"Within 1σ: {'YES' if within else 'NO'}, "
                    f"Within 2σ: {'YES' if within2 else 'NO'}, "
                    f"Error = {error_std:.4f} ev")
        else : 
            cs_loo = CubicSplineWithUncertainty(mu_ph[mask], peak_e[mask], x_err=mu_ph_sig[mask], y_err=e_unc[mask])
            pred = cs_loo(mu_ph[i])
            pred_mean = cs_loo.mean(mu_ph[i])
            pred_std = cs_loo.uncertainty(mu_ph[i])
            theor = peak_e[i]
            within = abs(theor - pred_mean) <= pred_std
            within2 = abs(theor - pred_mean) <= 2*pred_std
            error_std = theor - pred_mean
            if verbose:
                print(f"Filtered peak {i}: PH={mu_ph[i]:.2f}, True={theor:.4f} ev, "
                    f"Predicted ={pred_mean:.4f} ± {pred_std:.4f} ev, "
                    f"Within 1σ: {'YES' if within else 'NO'}, "
                    f"Within 2σ: {'YES' if within2 else 'NO'}, "
                    f"Error = {error_std:.4f} ev")
                
        pred_list.append(pred)
        pred_mean_list.append(pred_mean)
        pred_std_list.append(pred_std)
        within_list.append(within)
        within2_list.append(within2)
        error_std_list.append(error_std)
        

    if do_plot and len(loo_indices) > 0:
        plt.figure(figsize=(8,5))
        plt.errorbar(mu_ph, pred_mean_list, yerr=pred_std_list, fmt='o', label='Predicted (with uncertainty)')
        plt.plot(mu_ph, np.array(peak_e), 'rx', label='True value')
        for i, ok in enumerate(within_list):
            if not ok:
                plt.plot(mu_ph[i], pred_mean_list[i], 'ko', markerfacecolor='none', markersize=12, label='Outside 1σ' if i==within_list.index(False) else "")
        for i, ok in enumerate(within2_list):
            if not ok:
                plt.plot(mu_ph[i], pred_mean_list[i], 'ro', markerfacecolor='none', markersize=15, label='Outside 2σ' if i==within2_list.index(False) else "")
        plt.xlabel('PH Value')
        plt.ylabel('Energy (ev)')
        plt.title('Leave-one-out test on splines\nPrediction vs True Value (used points)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return pred_mean_list, pred_std_list, within_list, within2_list, error_std_list

def loo_gain(
    calib_output, peak_e, e_unc = None, do_plot=True, verbose=True, use_mask=False
):
    """
    Leave-one-out study for CubicSplineWithUncertainty with gain.
    For each calibration point that was used in the original fit, fit a spline leaving it out,
    predict its energy, and check if the true value is within the 1σ error band.
    Optionally plots the results.

    Parameters:
    - calib_output: dict, output from calibration()
    - peak_e: array-like, true energy values for each peak
    - do_plot: bool, plot the results
    - verbose: bool, print details
    - used_mask: array-like of bool, same length as peak_e. True for points used in the original spline.
    """
    calib_output = calib_output
    peak_e = peak_e
    mu_ph = np.array(calib_output["mu_ph"])
    mu_ph_sig = np.array(calib_output["mu_ph_sig"])
    n = len(mu_ph)

    if use_mask:
        used_mask = np.array(calib_output["mask_sig"])
    else :
        used_mask = np.ones(n, dtype=bool)

    pred_list = []
    pred_mean_list = []
    pred_std_list = []
    within_list = []
    within2_list = []
    error_std_list = []
    loo_indices = np.where(used_mask)[0]
    e_unc = e_unc

    for i in range(n):
        mask = used_mask.copy()
        if i in loo_indices:
            mask[i] = False  # leave this one out
            cs_loo = CubicSplineWithUncertainty(mu_ph[mask], peak_e[mask], x_err=mu_ph_sig[mask], y_err=e_unc[mask])
            pred = cs_loo.gain(mu_ph[i])
            pred_mean = cs_loo.g_mean(mu_ph[i])
            pred_std = cs_loo.g_uncertainty(mu_ph[i])
            theor = mu_ph[i]/peak_e[i]
            within = abs(theor - pred_mean) <= pred_std
            within2 = abs(theor - pred_mean) <= 2*pred_std
            error_std = theor - pred_mean
            if verbose:
                print(f"Removed peak {i}: PH={mu_ph[i]:.2f}, True={theor:.4f} PH/ev, "
                    f"Predicted ={pred_mean:.4f} ± {pred_std:.4f} PH/ev, "
                    f"Within 1σ: {'YES' if within else 'NO'}, "
                    f"Within 2σ: {'YES' if within2 else 'NO'}, "
                    f"Error = {error_std:.4f} PH/ev")
        else : 
            cs_loo = CubicSplineWithUncertainty(mu_ph[mask], peak_e[mask], x_err=mu_ph_sig[mask], y_err=e_unc[mask])
            pred = cs_loo.gain(mu_ph[i])
            pred_mean = cs_loo.g_mean(mu_ph[i])
            pred_std = cs_loo.g_uncertainty(mu_ph[i])
            theor = mu_ph[i]/peak_e[i]
            within = abs(theor - pred_mean) <= pred_std
            within2 = abs(theor - pred_mean) <= 2*pred_std
            error_std = theor - pred_mean
            if verbose:
                print(f"Removed peak {i}: PH={mu_ph[i]:.2f}, True={theor:.4e} PH/ev, "
                    f"Predicted ={pred_mean:.4e} ± {pred_std:.4e} PH/ev, "
                    f"Within 1σ: {'YES' if within else 'NO'}, "
                    f"Within 2σ: {'YES' if within2 else 'NO'}, "
                    f"Error = {error_std:.4e} PH/ev")
                
        pred_list.append(pred)
        pred_mean_list.append(pred_mean)
        pred_std_list.append(pred_std)
        within_list.append(within)
        within2_list.append(within2)
        error_std_list.append(error_std)
        

    if do_plot and len(loo_indices) > 0:
        plt.figure(figsize=(8,5))
        plt.errorbar(mu_ph, pred_mean_list, yerr=pred_std_list, fmt='o', label='Predicted (with uncertainty)')
        plt.plot(mu_ph, np.array(peak_e), 'rx', label='True value')
        for i, ok in enumerate(within_list):
            if not ok:
                plt.plot(mu_ph[i], pred_mean_list[i], 'ko', markerfacecolor='none', markersize=12, label='Outside 1σ' if i==within_list.index(False) else "")
        for i, ok in enumerate(within2_list):
            if not ok:
                plt.plot(mu_ph[i], pred_mean_list[i], 'ro', markerfacecolor='none', markersize=15, label='Outside 2σ' if i==within2_list.index(False) else "")
        plt.xlabel('PH Value')
        plt.ylabel('Gain (PH/ev)')
        plt.title('Loo test\nPrediction vs True Value (used points)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return pred_mean_list, pred_std_list, within_list, within2_list, error_std_list

class LooTest:
    """
    Leave-One-Out (LOO) test for TES calibration.
    Computes LOO results for all calibrated channels and provides a plot method.
    """
    def __init__(self, calibrated_ch, output, peak_e, e_unc, cal_space="energy"):
        """
        Parameters:
        - calibrated_ch: list of calibrated channel numbers
        - output: dict of calibration outputs per channel
        - peak_e: array of calibration energies
        - e_unc: array of calibration energy uncertainties
        - loo_func: function to perform leave-one-out (e.g., tes.loo)
        """
        allowed_cal_space = ["energy", "gain"]
        if cal_space not in allowed_cal_space:
            raise ValueError("Got unexpected value for the calibration space, value is either \"energy\" or \"gain\"")
        if cal_space == "energy" :
            loo_func = loo
            self.unit = "eV"
        if cal_space == "gain" :
            loo_func = loo_gain
            self.unit = "PH/eV"
        self.cal_space = cal_space
        self.calibrated_ch = calibrated_ch
        self.peak_e = peak_e
        self.results = {}
        for ch in calibrated_ch:
            calib_output = output[ch]
            pred_mean, pred_std, within, within2, error_std = loo_func(
                calib_output, peak_e, e_unc=e_unc, do_plot=False
            )
            self.results[ch] = {
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "within": within,
                "within2": within2,
                "error_std": error_std,
            }

    def plot(self, max_per_fig=7, ylim=None, use_idx= False):
        """
        Plot the difference between predicted and mean for each peak, for all channels.
        """
        batch = 0
        plt.figure(figsize=(9,6))
        for ch in self.calibrated_ch:
            if use_idx:
                xlabel = "Peak index"
                shift = batch * 0.05
            else :
                xlabel = "Peak energies (in eV)"
                shift = batch 
            batch += 1
            e_diff = np.array(self.results[ch]["error_std"])
            pred_std = np.array(self.results[ch]["pred_std"])
            idx_list = np.arange(1, len(self.peak_e)+1)
            if use_idx :
                plt.errorbar(idx_list + shift, e_diff, yerr=pred_std, fmt="o", label=f"Channel {ch}")
                plt.xticks(idx_list)
            else :
                plt.errorbar(self.peak_e + shift, e_diff, yerr=pred_std, fmt="o", label=f"Channel {ch}")
            if batch >= max_per_fig:
                batch = 0
                plt.legend()
                plt.xlabel(xlabel)
                plt.ylabel(f"{self.cal_space} error (in {self.unit})")
                plt.ylim(ylim)
                plt.axhline(0, linestyle="--", color="r")
                plt.title("Difference between predicted and mean for each peak")
                plt.show()
                plt.figure(figsize=(9,6))
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(f"{self.cal_space} error ({self.unit})")
        plt.ylim(ylim)
        if use_idx:
            plt.xticks(idx_list)
        plt.axhline(0, linestyle="--", color="r")
        plt.title("Difference between predicted and mean for each peak")
        plt.show()


def coadd_energy_histograms(data, chans, energy_range=(20, 120), nbins_e=6000, do_plot=True):
    """
    Coadd energy histograms for a list of channels.

    Parameters:
    - data: MASS TESGroup object
    - chans: list of channel numbers
    - energy_range: tuple, (min, max) energy in ev
    - nbins_e: int, number of bins
    - plot: bool, whether to plot the result

    Returns:
    - ebins: bin edges (array)
    - coaddE: shape (2, nbins_e), [all events, good events]
    """
    nbins_e = int(nbins_e)
    all_histos_E = np.zeros((2, len(chans), nbins_e))
    coaddE = np.zeros((2, nbins_e))

    for nn, ch in enumerate(chans):
        ds = data.channel[ch]
        g = ds.good()
        hE, bE = np.histogram(ds.p_energy[:], bins=nbins_e, range=energy_range)
        hEg, bEg = np.histogram(ds.p_energy[g], bins=nbins_e, range=energy_range)
        all_histos_E[0, nn, :] = hE
        all_histos_E[1, nn, :] = hEg
        coaddE[0, :] += hE
        coaddE[1, :] += hEg

    ebins = bE

    if do_plot:
        plt.figure(figsize=(13, 9))
        plt.semilogy(ebins[:-1], coaddE[0, :], label='all events')
        plt.semilogy(ebins[:-1], coaddE[1, :], label='all good')
        plt.xlim(energy_range)
        plt.xlabel('energy (ev)')
        plt.legend()
        plt.title('coadded energy spectrum')
        plt.show()

    return ebins, coaddE, all_histos_E

def plot_offset_calibrated_spectra(chans, all_histos_E, ebins, offset_base=1.3, energy_xlim=(1, 110), do_log=True):
    """
    Plot all calibrated energy spectra for each channel on the same graph with an offset.

    Parameters:
    - data: MASS TESGroup object
    - chans: list of channel numbers (same order as all_histos_E)
    - all_histos_E: np.ndarray, shape (2, n_channels, n_bins), histograms [all, good]
    - ebins: np.ndarray, energy bin edges
    - offset_base: float, base for offset scaling (default 1.3)
    - energy_xlim: tuple, x-axis limits for energy (default (1e3, 100e3))
    """

    plt.figure(figsize=(15, 9))
    for nn, ch in enumerate(chans):
        offset = offset_base ** (nn*15)
        plt.plot(
            ebins[:-1],
            (all_histos_E[1, nn, :] + 1) * offset,
            label=f"Ch {ch}"
        )

    plt.xlim(energy_xlim)
    plt.xlabel('energy (ev)')
    if do_log:
        plt.yscale('log')
    plt.title('Offset energy spectra by channel')
    plt.legend()
    plt.show()

# def kde(x_eval, mu, std, do_plot=True):
#     total = np.zeros_like(x_eval)
#     for mu, std in zip(mu, std):
#         total += gaussian(x_eval, 1, mu, std, 0)
#     if do_plot:
#         plt.figure(figsize=(13,9))
#         plt.plot(x_eval, total)
#         plt.xlabel("Energy (ev)")
#         plt.title("Kernel Density Estimation of the Energy")
#         plt.show()
#     return total

def kde(x_eval, mu, std, do_plot=True, log_scale = True):
    total = np.zeros_like(x_eval)
    for m, s in zip(mu, std):
        if np.isnan(m) or np.isnan(s) or s <= 0:
            continue  # skip invalid values
        total += gaussian(x_eval, 1, m, s, 0)
    if do_plot:
        plt.figure(figsize=(13,9))
        plt.plot(x_eval, total)
        if log_scale:
            plt.yscale("log")
        plt.xlabel("Energy (ev)")
        plt.title("Kernel Density Estimation of the Energy")
        plt.show()
    return total