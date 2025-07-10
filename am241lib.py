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

    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=p0)
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
        plt.axvline(mu, color='g', linestyle='--', label=f'Peak center {mu:.2f} ADC')
        plt.axvspan(mu - fwhm/2, mu + fwhm/2, color='orange', alpha=0.2, label='FWHM')
        plt.ylabel('Counts')
        if in_energy:
            plt.xlabel('Energy (keV)')
            plt.title(f'Gaussian Fit: FWHM = {fwhm:.2f} keV')
        else:
            plt.xlabel('ADC values')
            plt.title(f'Gaussian Fit: FWHM = {fwhm:.2f} ADC')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if return_sig:
        sigmas = np.sqrt(np.diag(pcov))
        return fwhm, popt, sigmas
    else :
        return fwhm, popt

'''
# def convolved_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2, c):
#     """
#     Convolution of two Gaussians is another Gaussian:
#     mean = mu1 + mu2
#     sigma = sqrt(sigma1^2 + sigma2^2)
#     amplitude = a1 * a2 * sqrt(2*pi*(sigma1^2 + sigma2^2))
#     """
#     mean = mu1 + mu2
#     sigma = np.sqrt(sigma1**2 + sigma2**2)
#     amplitude = a1 * a2 * np.sqrt(2 * np.pi * (sigma1**2 + sigma2**2))
#     return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + c

# def fit_convolved_gaussian(x, y, plot=True, in_energy=False):
#     """
#     Fit the convolution of two Gaussians to the given spectrum and return the FWHM.

#     Parameters:
#     - x: array-like, bin centers (e.g., energy)
#     - y: array-like, counts
#     - plot: bool, if True, plot the fit

#     Returns:
#     - fwhm: Full Width at Half Maximum of the convolved Gaussian
#     - popt: optimal parameters of the fit (a1, mu1, sigma1, a2, mu2, sigma2, c)
#     """
#     # Initial guess
#     a1 = np.max(y) - np.min(y)
#     mu1 = x[np.argmax(y)] / 2
#     sigma1 = (x[-1] - x[0]) / 20
#     a2 = a1/2
#     mu2 = x[np.argmax(y)] / 2
#     sigma2 = sigma1
#     c0 = np.min(y)
#     p0 = [a1, mu1, sigma1, a2, mu2, sigma2, c0]

#     try:
#         popt, pcov = curve_fit(convolved_gaussian, x, y, p0=p0, maxfev=10000)
#     except RuntimeError:
#         print("Convolved Gaussian fit did not converge.")
#         return None, None

#     # FWHM of the convolved Gaussian
#     sigma = np.sqrt(popt[2]**2 + popt[5]**2)
#     fwhm = 2.3548 * abs(sigma)

#     if plot:
#         plt.figure(figsize=(6,4))
#         plt.plot(x, y, 'b.', label='Data')
#         xfit = np.linspace(np.min(x), np.max(x), 500)
#         plt.plot(xfit, convolved_gaussian(xfit, *popt), 'r-', label='Convolved Gaussian fit')
#         mean = popt[1] + popt[4]
#         plt.axvline(mean, color='g', linestyle='--', label=f'Peak center {mean:.2f}')
#         plt.axvspan(mean - fwhm/2, mean + fwhm/2, color='orange', alpha=0.2, label='FWHM')
#         plt.ylabel('Counts')
#         if in_energy:
#             plt.xlabel('Energy (keV)')
#             plt.title(f'Convolved Gaussian Fit: FWHM = {fwhm:.2f} keV')
#         else:
#             plt.xlabel('ADC values')
#             plt.title(f'Convolved Gaussian Fit: FWHM = {fwhm:.2f} ADC')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     return fwhm, popt
'''

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
            plt.xlabel('Energy (keV)')
            plt.title(f'Double Gaussian Fit: FWHM1 = {fwhm1:.2f} keV, FWHM2 = {fwhm2:.2f} keV')
        else:
            plt.xlabel('ADC values')
            plt.title(f'Double Gaussian Fit: FWHM1 = {fwhm1:.2f} ADC, FWHM2 = {fwhm2:.2f} ADC')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if return_sig:
        sigmas = np.sqrt(np.diag(pcov))
        return (fwhm1, fwhm2), popt, sigmas
    else :
        return (fwhm1, fwhm2), popt

class Coadd_data:
    def __init__(self, data, timebase, calibration_output, ch_list):
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
    
    def hist(self, energy_range=(20,120), nbins = 60000, do_plot = True, figsize = (9,4)):
        """
        Calculates the histogram of the coadded data

        Parameters:
        - energy_range: tuple, (min, max) energy in keV
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
            plt.xlim(energy_range)
            plt.xlabel('energy (keV)')
            plt.legend()
            plt.title('coadded energy spectrum')
            plt.show()
        return [c, cg], [b, bg]
    
    def plot_offset_spectra(self, energy_range=(20, 120), nbins=6000, offset_base=1.3, do_log=True):
        """
        Plot all calibrated energy spectra for each channel in the Coadd_data object on the same graph with an offset.

        Parameters:
        - energy_range: tuple, x-axis limits for energy (default (1, 110))
        - nbins: int, number of bins for the histogram
        - offset_base: float, base for offset scaling (default 1.3)
        - do_log: bool, whether to use log scale for y-axis
        """
        plt.figure(figsize=(15, 9))
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
        plt.xlabel('energy (keV)')
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
        self.splines = [CubicSpline(x_perturbed[i], y_perturbed[i]) for i in range(self.n_samples)]
        self.cs = CubicSpline(self.x, self.y)

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
            plt.xlabel("Possible energy (keV)")
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
    
    def return_paramater(self):
        return self.a, self.b, self.a_err, self.b_err



def horizontal_distances_to_neighbors(points_x, idx):
    """
    Calculate the horizontal distances from the removed calibration point
    to its two neighboring calibration points.

    Parameters:
    - points_x: array-like, ADC values of calibration points (must be 1D)
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
        ax.xlabel("ADC values")
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
    peak_adc = 0.5 * (bin_edges[peak_index] + bin_edges[peak_index + 1])
    peak_count = counts[peak_index]
    return peak_index, peak_adc, peak_count

def get_bin_count(adc_value, counts, bin_edges):
    bin_index = np.searchsorted(bin_edges, adc_value, side='right') - 1
    if bin_index < 0 or bin_index >= len(counts):
        return 0
    return counts[bin_index]

# def zoom_in_on_peaks(ds, peak_data, peak_index=0, zoom_factor=0.1, zbins=200, hheight = True):
#     g = ds.good()
#     values = ds.p_filt_value_dc[g]
#     ch = ds.channum
#     # counts, bin_edges = np.histogram(values, bins=bins, range=hist_range)
#     # log_counts = np.log(counts + 1)
#     # peaks, _ = find_peaks(log_counts, height=0, distance=30, prominence=2.5)
#     # peak_heights = counts[peaks]
#     # sorted_indices = np.argsort(peak_heights)[::-1]
#     # if len(sorted_indices) == 0:
#     #     print("No peaks found.")
#     #     return None, None
#     peak_idx_list = peak_data[0][ch]
#     count_list = peak_data[1][ch]
#     adc_list = peak_data[2][ch]
    


#     sorted_indices = np.argsort(count_list)[::-1]

#     chosen_peak_idx = peak_index
#     chosen_peak_count = adc_list[chosen_peak_idx]

#     window = 1000
#     zoom_min = chosen_peak_count - window*zoom_factor
#     zoom_max = chosen_peak_count + window*zoom_factor
#     mask = (values >= zoom_min) & (values <= zoom_max)
#     zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))
#     zoom_peak_height = get_bin_count(chosen_peak_count, zoom_counts, zoom_bin_edges)
#     half_height = zoom_peak_height / 2
#     plt.figure(figsize=(6, 4))
#     plt.plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
#     plt.axvline(chosen_peak_count, color='red', linestyle='--', label='Selected peak', alpha = 0.3)
#     if hheight :
#         plt.axhline(half_height, color='green', linestyle='--', label='Half height', alpha = 0.3)
#     plt.xlabel("ADC Value")
#     plt.ylabel("Counts")
#     plt.title(f"Zoom on peak {peak_index} at {chosen_peak_count:.1f}")
#     plt.ylim(0, 1.2*zoom_peak_height)
#     plt.xlim(zoom_min, zoom_max)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     return zoom_counts, zoom_bin_edges

def zoom_in_on_peaks(ds, peak_data, peak_index=0, zoom_factor=0.1, zbins=200, hheight=True):
    """
    Display the zoomed-in peak and, next to it, the full spectrum with the selected peak highlighted.

    Parameters:
    - ds: MASS dataset for a channel
    - peak_data: list of [calp_dict, count_dict, adc_val_dict]
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
    adc_list = peak_data[2][ch]

    chosen_peak_idx = peak_index
    chosen_peak_count = adc_list[chosen_peak_idx]
    chosen_peak_height = count_list[chosen_peak_idx]

    window = 1000
    zoom_min = chosen_peak_count - window * zoom_factor
    zoom_max = chosen_peak_count + window * zoom_factor
    mask = (values >= zoom_min) & (values <= zoom_max)
    zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))
    zoom_peak_height = get_bin_count(chosen_peak_count, zoom_counts, zoom_bin_edges)
    half_height = zoom_peak_height / 2

    # --- Plotting ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Full spectrum with all peaks, highlight selected
    full_counts, full_bins = np.histogram(values, bins=6000, range=(0, 4e3))
    axs[0].plot(full_bins[:-1], full_counts, label="Full good spectrum")
    axs[0].plot(chosen_peak_count, chosen_peak_height, 'rx', label='Selected peak')
    axs[0].set_xlabel("ADC Value")
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
    axs[1].set_xlabel("ADC Value")
    axs[1].set_ylabel("Counts")
    axs[1].set_title(f"Zoom on peak {peak_index} at {chosen_peak_count:.1f}")
    axs[1].set_ylim(0, 1.2 * zoom_peak_height)
    axs[1].set_xlim(zoom_min, zoom_max)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    return zoom_counts, zoom_bin_edges

# def zoom_in_on_peaks_e(ds, peak_data, peak_index, zoom_factor=0.1, zbins=200, hheight = True):
#     g = ds.good()
#     values = ds.p_energy[g]
#     ch = ds.channum
#     # counts, bin_edges = np.histogram(values, bins=bins, range=hist_range)
#     # log_counts = np.log(counts + 1)
#     # peaks, _ = find_peaks(log_counts, height=0, distance=30, prominence=2.5)
#     # peak_heights = counts[peaks]
#     # sorted_indices = np.argsort(peak_heights)[::-1]
#     # if len(sorted_indices) == 0:
#     #     print("No peaks found.")
#     #     return None, None
#     peak_idx_list = peak_data[0][ch]
#     count_list = peak_data[1][ch]
#     adc_list = peak_data[2][ch]
#     energy_list = peak_data[3][ch]
    


#     sorted_indices = np.argsort(count_list)[::-1]

#     chosen_peak_idx = peak_index
#     chosen_peak_count = energy_list[chosen_peak_idx]

#     window = 20
#     zoom_min = chosen_peak_count - window*zoom_factor
#     zoom_max = chosen_peak_count + window*zoom_factor
#     mask = (values >= zoom_min) & (values <= zoom_max)
#     zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))
#     zoom_peak_height = get_bin_count(chosen_peak_count, zoom_counts, zoom_bin_edges)
#     half_height = zoom_peak_height / 2
#     plt.figure(figsize=(6, 4))
#     plt.plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
#     plt.axvline(chosen_peak_count, color='red', linestyle='--', label='Selected peak', alpha = 0.3)
#     if hheight :
#         plt.axhline(half_height, color='green', linestyle='--', label='Half height', alpha = 0.3)
#     plt.xlabel("ADC Value")
#     plt.ylabel("Counts")
#     plt.title(f"Zoom on peak {peak_index} at {chosen_peak_count:.1f} keV")
#     plt.ylim(0, 1.2*zoom_peak_height)
#     plt.xlim(zoom_min, zoom_max)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     return zoom_counts, zoom_bin_edges

def zoom_in_on_peaks_e(ds, peak_data, peak_index, zoom_factor=0.1, zbins=200, hheight=True, do_plot=True):
    """
    Display the zoomed-in energy peak and, next to it, the full energy spectrum with the selected peak highlighted.

    Parameters:
    - ds: MASS dataset for a channel
    - peak_data: list of [calp_dict, count_dict, adc_val_dict, energy_dict]
    - peak_index: which peak to zoom in on
    - zoom_factor: relative width of zoom window
    - zbins: number of bins for zoomed histogram
    - hheight: whether to plot the half-height line
    """

    g = ds.good()
    values = ds.p_energy[g]
    ch = ds.channum

    peak_idx_list = peak_data[0][ch]
    count_list = peak_data[1][ch]
    adc_list = peak_data[2][ch]
    energy_list = peak_data[3][ch]

    chosen_peak_idx = peak_index
    chosen_peak_energy = energy_list[chosen_peak_idx]
    chosen_peak_height = count_list[chosen_peak_idx]

    window = 20
    zoom_min = chosen_peak_energy - window * zoom_factor
    zoom_max = chosen_peak_energy + window * zoom_factor
    mask = (values >= zoom_min) & (values <= zoom_max)
    zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))
    zoom_peak_height = get_bin_count(chosen_peak_energy, zoom_counts, zoom_bin_edges)
    half_height = zoom_peak_height / 2

    if do_plot:
        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Full energy spectrum with all peaks, highlight selected
        full_counts, full_bins = np.histogram(values, bins=6000, range=(0, 150))
        axs[0].plot(full_bins[:-1], full_counts, label="Full good spectrum")
        axs[0].plot(chosen_peak_energy, chosen_peak_height, 'rx', label='Selected peak')
        axs[0].set_xlabel("Energy (keV)")
        axs[0].set_ylabel("Counts")
        axs[0].set_title(f"Full energy spectrum (ch {ch})")
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[0].set_xlim(full_bins[0], full_bins[-1])

        # Right: Zoomed-in peak
        axs[1].plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
        axs[1].axvline(chosen_peak_energy, color='red', linestyle='--', label='Selected peak', alpha=0.3)
        if hheight:
            axs[1].axhline(half_height, color='green', linestyle='--', label='Half height', alpha=0.3)
        axs[1].set_xlabel("Energy (keV)")
        axs[1].set_ylabel("Counts")
        axs[1].set_title(f"Zoom on peak {peak_index} at {chosen_peak_energy:.1f} keV")
        axs[1].set_ylim(0, 1.2 * zoom_peak_height)
        axs[1].set_xlim(zoom_min, zoom_max)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    return zoom_counts, zoom_bin_edges

def zoom_in_adc(ds, adc_value, zoom_factor=0.1, zbins=200, do_plot = True):
    """
    Zoom in on a given ADC value and show the zoomed histogram and the full spectrum.

    Parameters:
    - ds: MASS dataset for a channel
    - adc_value: ADC value to center the zoom on
    - zoom_factor: relative width of zoom window (fraction of 1000)
    - zbins: number of bins for zoomed histogram
    - hheight: whether to plot the half-height line
    """
    g = ds.good()
    values = ds.p_filt_value_dc[g]
    ch = ds.channum

    window = 1000
    zoom_min = adc_value - window * zoom_factor
    zoom_max = adc_value + window * zoom_factor
    mask = (values >= zoom_min) & (values <= zoom_max)
    zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))

    if do_plot:
        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Full spectrum with the selected ADC value highlighted
        full_counts, full_bins = np.histogram(values, bins=6000, range=(0, 4e3))
        axs[0].plot(full_bins[:-1], full_counts, label="Full good spectrum")
        axs[0].axvline(adc_value, color='red', linestyle='--', label='Selected ADC')
        axs[0].set_xlabel("ADC Value")
        axs[0].set_ylabel("Counts")
        axs[0].set_title(f"Full spectrum (ch {ch})")
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[0].set_xlim(full_bins[0], full_bins[-1])

        # Right: Zoomed-in region
        axs[1].plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
        axs[1].axvline(adc_value, color='red', linestyle='--', label='Selected ADC', alpha=0.3)
        axs[1].set_xlabel("ADC Value")
        axs[1].set_ylabel("Counts")
        axs[1].set_title(f"Zoom on ADC {adc_value:.1f}")
        axs[1].set_xlim(zoom_min, zoom_max)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    return zoom_counts, zoom_bin_edges

def zoom_in_energy(ds, energy_value, zoom_factor=0.1, zbins=200, hheight=True, do_plot= True):
    """
    Zoom in on a given Energy value and show the zoomed histogram and the full energy spectrum.

    Parameters:
    - ds: MASS dataset for a channel
    - energy_value: Energy value (keV) to center the zoom on
    - zoom_factor: relative width of zoom window (fraction of 20)
    - zbins: number of bins for zoomed histogram
    - hheight: whether to plot the half-height line
    """
    g = ds.good()
    values = ds.p_energy[g]
    ch = ds.channum

    window = 20
    zoom_min = energy_value - window * zoom_factor
    zoom_max = energy_value + window * zoom_factor
    mask = (values >= zoom_min) & (values <= zoom_max)
    zoom_counts, zoom_bin_edges = np.histogram(values[mask], bins=zbins, range=(zoom_min, zoom_max))

    if do_plot:
        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Full energy spectrum with the selected energy highlighted
        full_counts, full_bins = np.histogram(values, bins=6000, range=(0, 150))
        axs[0].plot(full_bins[:-1], full_counts, label="Full good spectrum")
        axs[0].axvline(energy_value, color='red', linestyle='--', label='Selected Energy')
        axs[0].set_xlabel("Energy (keV)")
        axs[0].set_ylabel("Counts")
        axs[0].set_title(f"Full energy spectrum (ch {ch})")
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[0].set_xlim(full_bins[0], full_bins[-1])

        # Right: Zoomed-in region
        axs[1].plot(zoom_bin_edges[:-1], zoom_counts, label="Zoomed histogram", color='blue')
        axs[1].axvline(energy_value, color='red', linestyle='--', label='Selected Energy', alpha=0.3)
        axs[1].set_xlabel("Energy (keV)")
        axs[1].set_ylabel("Counts")
        axs[1].set_title(f"Zoom on Energy {energy_value:.1f} keV")
        axs[1].set_xlim(zoom_min, zoom_max)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    return zoom_counts, zoom_bin_edges

"""
# def calibrate_channel_with_escape_peaks(data, ch, peak_data, Timebase=None, zoom_factor=0.05, bc_type='natural', escpk_idx=0):
#     
#     Calibrate a channel using escape peaks, following the steps for channel 45.
#     Plots all relevant figures and prints key results.

#     Parameters:
#     - data: MASS TESGroup object
#     - ch: channel number (int)
#     - peak_val: list of reference energies (keV)
#     - zoom_factor: float, zoom factor for escape peak search
#     

#     ds = data.channel[ch]
#     g = ds.good()


#     calp = np.copy(peak_data[0][ch])   # The index of the peaks used for calibration
#     count = np.copy(peak_data[1][ch])  # The count for each peak used for calibration
#     adc_val = np.copy(peak_data[2][ch])  # The ADC value for each peak used for calibration
#     peak_val = np.copy(peak_data[3])    # The energy value for each peak
#     p_mu = np.emptylike(peak_val)       # The mean values of the distribution of each peak (without the escape peaks)
#     p_sigma = np.emptylike(peak_val)    # The sigma values of the distribution of each peak (without the escape peaks)

#     num_none = peak_val.count(None)  # Counts to see if there are 4 peaks or 2 peaks due to the k alphas of tin

#     # Plotting the ADC values spectrum
#     plt.figure(figsize=(5, 3.5))
#     add_channel_histogram_to_ax(data, ch, plt, do_allg=False, do_peaks=False, Timebase=Timebase)
#     plt.plot(peak_data[2][ch], peak_data[1][ch], "rx", label="peaks")
#     plt.show()


#     print(f"The index of the peaks for channel {ch} are : {peak_data[0][ch]}")
#     print(f"The counts of the peaks for channel {ch} are : {peak_data[1][ch]}")
#     print(f"The ADC values of the peaks for channel {ch} are : {peak_data[2][ch]}")

#     # # Zoom in on the first peak (index=0)
#     # czoom_f, bzoom_f = zoom_in_on_peaks(ds, peak_data, peak_index=0, zoom_factor=0.07, zbins=50, hheight=False)
#     # fwhm, popt = fit_fwhm(x=bzoom_f[:-1], y=czoom_f)
#     # a, mu_f, sigma, c = popt
#     # print(f"The center of the peak is at {mu_f} ADC")


#     # # Zoom in on the main peak (third peak, index=2)
#     # czoom_m, bzoom_m = zoom_in_on_peaks(ds, peak_data, peak_index=2, zoom_factor=0.005, zbins=400, hheight=False)
#     # fwhm, popt = fit_fwhm(x=bzoom_m[:-1], y=czoom_m)
#     # a, mu_m, sigma, c = popt
#     # print(f"The center of the peak is at {mu_m} ADC")
#     i_shift = 0
#     for i in range(len(peak_val)):
#         n = i - i_shift
#         if peak_val[i] == None and peak_val[i+1] == None:
#             czoom, bzoom = zoom_in_on_peaks(ds, peak_data=peak_data, peak_index=n, zoom_factor=zoom_factor, zbins=100)
#             bpeak_idx, bpeak_adc, bpeak_count = find_biggest_peak(czoom, bzoom)
#             min_height = int(bpeak_count * 0.4)
#             max_height = int(bpeak_count * 0.9)
#             peaks_zoom, properties_zoom = find_peaks(czoom, height=(min_height, max_height))
#             plt.figure(figsize=(6, 4))
#             plt.plot(bzoom[:-1], czoom, label="Zoomed histogram", color='blue')
#             plt.plot(bzoom[:-1][peaks_zoom], czoom[peaks_zoom], "rx", label="Peaks", markersize=5)
#             plt.axhline(min_height, color="red", linestyle="--", label="Min height", alpha=0.3)
#             plt.axhline(max_height, color="green", linestyle="--", label="Max height", alpha=0.3)
            
#             plt.xlabel("ADC Value")
#             plt.ylabel("Counts")
#             plt.title("Zoomed Histogram with Peaks")
#             plt.xlim(min(bzoom[:-1]), max(bzoom[:-1]))
#             plt.legend()
#             plt.tight_layout()
#             plt.show()
#             escape_peaks = [bpeak_adc, bzoom[:-1][peaks_zoom[escpk_idx[i_shift]]]]
#             i_shift += 1
#             p_mu[i] = None
#             p_sigma[i] = None
#         else :
#             czoom_f, bzoom_f = zoom_in_on_peaks(ds, peak_data, peak_index=n, hheight=False)
#             pk_adc = adc_val[n]
#             adc_delta = 20
#             bin_mask = (bzoom_f>pk_adc-adc_delta) & (bzoom_f<pk_adc+adc_delta)
#             fwhm, popt = fit_fwhm(x=bzoom_f[bin_mask], y=czoom_f[bin_mask])
#             a, p_mu[i], p_sigma[i], c = popt
#             print(f"The center of the peak {n} is at {p_mu[i]} pm {np.sqrt(p_sigma[i]):.2f}ADC")


#     # Zoom in on the second peak (second peak, index=1)
#     czoom, bzoom = zoom_in_on_peaks(ds, peak_data=peak_data, peak_index=1, zoom_factor=zoom_factor, zbins=100)
#     bpeak_idx, bpeak_adc, bpeak_count = find_biggest_peak(czoom, bzoom)
#     min_height = int(bpeak_count * 0.4)
#     max_height = int(bpeak_count * 0.9)
#     peaks_zoom, properties_zoom = find_peaks(czoom, height=(min_height, max_height))
#     plt.figure(figsize=(6, 4))
#     plt.plot(bzoom[:-1], czoom, label="Zoomed histogram", color='blue')
#     plt.plot(bzoom[:-1][peaks_zoom], czoom[peaks_zoom], "rx", label="Peaks", markersize=5)
#     plt.axhline(min_height, color="red", linestyle="--", label="Min height", alpha=0.3)
#     plt.axhline(max_height, color="green", linestyle="--", label="Max height", alpha=0.3)
    
#     plt.xlabel("ADC Value")
#     plt.ylabel("Counts")
#     plt.title("Zoomed Histogram with Peaks")
#     plt.xlim(min(bzoom[:-1]), max(bzoom[:-1]))
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     escape_peaks = [bpeak_adc, bzoom[:-1][peaks_zoom[escpk_idx]]] 

#     # --- Prepare calibration points with escape peaks ---
#     cal_peaks_ch = np.copy(peak_data[2][ch])
#     cal_peaks_ch = np.delete(cal_peaks_ch, 1)  # Remove the second peak
#     cal_peaks_ch = np.insert(cal_peaks_ch, 1, escape_peaks)
#     cal_peaks_ch[0] = mu_f
#     cal_peaks_ch[3] = mu_m
#     ka1_e = 25.27136 # in keV
#     ka2_e = 25.04404 # in keV
#     esc_p1e = peak_val[2] - ka1_e
#     esc_p2e = peak_val[2] - ka2_e
#     print(f"The energies of the escape peaks are : {[esc_p1e, esc_p2e]} keV")
#     peak_val_ch = np.copy(peak_val)
#     peak_val_ch = np.delete(peak_val_ch, 1)
#     peak_val_ch = np.insert(peak_val_ch, 1, [esc_p1e, esc_p2e])

#     print(f"The energy values for the calibration peaks are : {peak_val_ch}")
#     print(f"The ADC values for the calibration peaks are : {cal_peaks_ch}")
#     print(f"The shape of the energy values is {peak_val_ch.shape}")
#     print(f"The shape of the ADC values is {cal_peaks_ch.shape}")


#     peaks = cal_peaks_ch
#     energies = peak_val_ch
#     cs_ch = CubicSpline(peaks, energies, bc_type = bc_type)

#     x_fit = np.linspace(0, 8000, 8000)
#     y_fit_cs = cs_ch(x_fit)

#     plt.figure(figsize=(5, 3.5))
#     plt.plot(peaks, energies, 'ro', label='Data Points', markersize=5)
#     plt.plot(x_fit, y_fit_cs, label='Interpolated Cubic Spline', linestyle='--')
#     plt.xlabel('ADC Values')
#     plt.ylabel('Energy (keV)')
#     plt.title(f'Channel {ch} Interpolated Calibration')  
#     plt.legend()
#     plt.xlim((0, 8e3))
#     plt.show()


#     # adc_points = peaks
#     # energy_points = energies
#     # cs_spreads = [y_spread(x, y, cs_ch) for x, y in zip(adc_points, energy_points)]

#     # plt.figure(figsize=(7, 4))
#     # plt.plot(adc_points, cs_spreads, 's-', label='Cubic Spline Fit Spread')
#     # plt.xlabel('ADC Value')
#     # plt.ylabel('Spread (keV)')
#     # plt.title(f'Spread between Calibration Points and Fit (Channel {ch})')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.show()

#     # # Print fit values at calibration points
#     # for adc, cal_energy in zip(peaks, energies):
#     #     energy = cs_ch(adc)
#     #     print(f"ADC: {adc:.1f}, Energy: {energy:.4f} keV (calibrated: {cal_energy:.4f} keV), Spread: {abs(energy - cal_energy):.4f} keV")

#     # --- Apply calibration and plot spectrum ---
#     mask = ds.p_filt_value_dc[:]>(cal_peaks_ch[-1] + 1000)
#     if np.any(mask):
#         print("At least one element is selected by the mask.")
#     else:   
#         print("No elements are selected by the mask.")  
#     print("Number of elements selected:", np.sum(mask))
#     ds.p_energy[:] = cs_ch(ds.p_filt_value_dc[:])
#     ds.p_energy[mask] = None
#     g = ds.good()
#     c_e, b_e = np.histogram(ds.p_energy[:], bins=6000, range=(0, max(ds.p_energy[:])))
#     cg_e, bg_e = np.histogram(ds.p_energy[g], bins=6000, range=(0, max(ds.p_energy[:])))
#     plt.figure(figsize=(5, 3.5))
#     plt.plot(b_e[:-1], c_e, label="all")
#     plt.plot(bg_e[:-1], cg_e, label="good")
#     plt.xlabel("Energy (keV)")
#     plt.ylabel("Counts")
#     plt.title(f'Channel {ch} Energy Spectrum (Cubic Spline Calibration)')
#     plt.legend()
#     plt.yscale("log")
#     plt.show()

#     # Return calibration objects for further use if needed
#     return {
#         "cal_peaks": peaks,
#         "peak_val": energies,
#         "cs": cs_ch,
#         "counts": [c_e, cg_e],
#         "bins": [b_e, bg_e]
#     }
"""

class Calibration :
    def __init__(self, data, peak_e, Timebase=None, escp_pk_e= None):
        self.data= data
        self.peak_e = peak_e
        self.timebase = Timebase
        self.escp_pk_e = escp_pk_e
        
    def __call__(self, ch, zoom_bins = 700, do_plot = True, filter_uncertainty = False):
        # Treating the inputed data
        ds = self.data.channel[ch]
        peak_e = self.peak_e
        escp_pk_e = self.escp_pk_e
        Timebase = self.timebase
        g = ds.good()
        adc = ds.p_filt_value_dc[:]
        zbins = zoom_bins

        if Timebase is not None:
            mask = Timebase[ch] > 100
            gmask =  mask & g
        else : gmask = g
        adc_filtered = adc[gmask]

        if escp_pk_e is not None :
            escp_pk_idx = []
            for e in escp_pk_e :
                escp_pk_idx.append(int(np.where(peak_e==e)[0][0]))
            print(f"The indices of the escape peaks are {escp_pk_idx}")
        

        figsize = (9,6)
        mpeak_e = 59.5409 # keV, Initializing the value for the main peak of 241am

        # Making a very discrete spectrum of the inputed data
        nbins = int(15e3)
        bin_range= (0,4e3)
        c, b = np.histogram(adc, bins =nbins, range = bin_range)
        cg, bg = np.histogram(adc_filtered, bins= nbins, range = bin_range)

        if do_plot:
            # Visualizing the spectrum
            plt.figure(figsize=figsize)
            plt.plot(b[:-1], c, label="all")
            plt.plot(bg[:-1],cg, label="filtered")
            plt.xlabel("ADC values")
            plt.ylabel("Comptes")
            plt.title(f'ch {ch}')
            plt.xlim(bin_range)
            plt.yscale("log")
            plt.legend()

        # Showing the adc value for the main peak
        mpeak_adc = bg[np.argmax(cg)]
        print(f"For the main peak ({mpeak_e} keV) we have a first approximative ADC value of {mpeak_adc} ADC \n")

        # Converting the list of given energy values to ADC
        kev_to_adc = mpeak_adc/mpeak_e
        print(f"The linear coefficient are :\n - {kev_to_adc} ADC/keV \n - {1/kev_to_adc} keV/ADC\n")
        peak_adc = peak_e*kev_to_adc

        # Finding the precise ADC values for each peak
        mu_adc = np.empty(len(peak_adc))
        mu_adc_sig = np.empty(len(peak_adc))
        sigma_adc = np.empty(len(peak_adc))
        skip = False
        for i in range(len(mu_adc)):
            if skip :
                skip = False
            elif i in escp_pk_idx:
                zc, zb = zoom_in_adc(ds, peak_adc[i]+5, zbins=zbins, do_plot=do_plot)
                adc_delta = 15
                max_count = 1
                while max_count<= 3:
                    zmask = (zb > peak_adc[i]-adc_delta+5) & (zb < peak_adc[i]+adc_delta+5)
                    zc_filt = zc[zmask[:-1]]
                    zb_filt = zb[zmask]
                    max_count = max(zc_filt)
                    adc_delta += 5
                fwhm, popt, sigmas = fit_double_gaussian(zb_filt, zc_filt, plot=do_plot, return_sig=True)
                a1, mu1, sigma1, a2, mu2, sigma2, c = popt
                a1_sig, mu1_sig, sigma1_sig, a2_sig, mu2_sig, sigma2_sig, c_sig = sigmas
                if mu1 < mu2:
                    mu_adc[i] = mu1
                    mu_adc[i+1] = mu2
                    sigma_adc[i] = sigma1
                    sigma_adc[i+1] = sigma2
                    mu_adc_sig[i] = mu1_sig
                    mu_adc_sig[i+1] = mu2_sig
                else : 
                    mu_adc[i+1] = mu1
                    mu_adc[i] = mu2
                    sigma_adc[i+1] = sigma1
                    sigma_adc[i] = sigma2
                    mu_adc_sig[i+1] = mu1_sig
                    mu_adc_sig[i] = mu2_sig
                skip = True
            else :
                if peak_adc[i] > 2000:
                    zbins = 400
                else :
                    zbins = zoom_bins
                zc, zb = zoom_in_adc(ds, peak_adc[i], zbins=zbins, do_plot=do_plot)
                adc_delta = 15
                max_count = 1
                zoom_tries = 0
                do_fit = True
                while max_count<= 2:
                    print(f"Zoom try n°{zoom_tries}, ")
                    zmask = (zb > peak_adc[i]-adc_delta) & (zb < peak_adc[i]+adc_delta)
                    zc_filt = zc[zmask[:-1]]
                    zb_filt = zb[zmask]
                    max_count = max(zc_filt)
                    adc_delta += 10
                    if zoom_tries > 4:
                        print(f"Couldn't find peak near {peak_adc[i]} ADC")
                        do_fit = False
                        break
                        # raise StopIteration(f"Couldn't find peak near {peak_adc[i]} ADC")
                    zoom_tries += 1
                    if zoom_tries> 1:
                        if do_plot:
                            plt.figure(figsize=figsize)
                            plt.plot(zb_filt, zc_filt, "bo")
                            plt.axvline(peak_adc[i], color="red", linestyle="--")
                            plt.title(f"Zoom on center {peak_adc[i]} ± {adc_delta} ADC")
                            plt.show()
                if do_fit:
                    fwhm, popt, sigmas = fit_fwhm(zb_filt, zc_filt, plot=do_plot, return_sig=True)
                    a, mu_adc[i], sigma_adc[i], c = popt
                    a_sig, mu_adc_sig[i], sigma_sig, c_sig = sigmas
                else :
                    mu_adc[i], sigma_adc[i], mu_adc_sig[i] = None, None, None

            # # Condition in case the fit targets the previous peak
            # last_adc = mu_adc[i-1]
            # shift = 0
            # tries = 0
            # success = True
            # adc_delta = 10
            # while (mu_adc[i] > last_adc-1) and (mu_adc[i]<last_adc+1):
            #     print(f"Number of tries : {tries+1}")
            #     if tries < 4:
            #         tries += 1
            #         shift = 5*tries
            #         zmask = (zb > peak_adc[i]-adc_delta+shift) & (zb < peak_adc[i]+adc_delta+shift)
            #         zc_filt = zc[zmask[:-1]]
            #         zb_filt = zb[zmask]
            #         fwhm, popt = fit_fwhm(zb_filt, zc_filt, plot=False)
            #         a, mu_adc[i], sigma_adc[i], c = popt
            #         # plt.figure(figsize=(6,4))
            #         # plt.plot(zb_filt, zc_filt, "bo")
            #         # plt.axvline(peak_adc[i]+shift)
            #         # plt.show()
            #     elif tries > 3 and tries < 7:
            #         tries += 1
            #         shift = -5*(tries-4)
            #         zmask = (zb > peak_adc[i]-adc_delta+shift) & (zb < peak_adc[i]+adc_delta+shift)
            #         zc_filt = zc[zmask[:-1]]
            #         zb_filt = zb[zmask]
            #         fwhm, popt = fit_fwhm(zb_filt, zc_filt, plot=False)
            #         a, mu_adc[i], sigma_adc[i], c = popt
            #     else :
            #         mu_adc[i] = None
            #         sigma_adc[i] = None
            #         print("The fit was not able to find a different peak")
            #         success = False
            #         break
                
            # if success:
            #     _1, _2 = fit_fwhm(zb_filt, zc_filt)

        print(f"The ADC values of the peaks are : {mu_adc}")
        print(f"The uncertainties for the peak's ADC values are :{mu_adc_sig}")

        # x_test = np.linspace(0, mu_adc[-1])
        # plt.errorbar(mu_adc, peak_e,xerr=sigma_adc, fmt="rx", )
        # plt.plot(x_test, x_test/kev_to_adc)
        # plt.title("ADC Values vs Energy in keV of the calibration points")
        # plt.xlabel("ADC values")
        # plt.ylabel("Energy (keV)")
        # # plt.xlim(800,1000)
        # plt.show()
        
        # Calibrating the gain curve with a cubic spline object
        cs = CubicSplineWithUncertainty(mu_adc, peak_e, x_err=mu_adc_sig)

        x_gain = np.linspace(mu_adc[0]-100, mu_adc[-1]+100, 1000)
        y_gain = cs(x_gain)                 # The energy value according to a simple cubicspline
        y_mean = cs.mean(x_gain)            # The mean energy value for the cubicspline with incertitudes
        y_std = cs.uncertainty(x_gain)      # The std energy value for the cubicspline with incertitudes

        # if do_plot:
        # Visualizing the gain curve
        plt.figure(figsize=figsize)
        plt.errorbar(mu_adc, peak_e,xerr=mu_adc_sig, fmt="rx")
        plt.plot(x_gain, y_gain, label = "Gain spline")
        plt.plot(x_gain, y_mean, "y--", label="Mean spline")
        plt.fill_between(x_gain, y_mean - y_std, y_mean + y_std, alpha=0.3, label='Uncertainty (1σ)')
        plt.legend()
        plt.xlabel('ADC Values')
        plt.ylabel('Energy values (keV)')
        plt.title(f'Gain curve of Channel {ch}')
        plt.show()


        # Visualizing the difference between the theorical value and the predicted energy
        upper_range = cs.mean(mu_adc)+cs.uncertainty(mu_adc)
        lower_range = cs.mean(mu_adc)-cs.uncertainty(mu_adc)
        if do_plot:
            plt.figure(figsize=figsize)
            plt.plot(mu_adc, peak_e-cs(mu_adc), "bo-", label="Theorical - gain")
            plt.plot(mu_adc, peak_e-cs.mean(mu_adc), "ro-", label="Theorical - mean")
            plt.fill_between(mu_adc,peak_e-lower_range, peak_e-upper_range, alpha=0.3, label='Uncertainty (1σ)')
            plt.xlabel("ADC Value")
            plt.ylabel("Theorical - Predicted Energy (keV)")
            plt.title("Difference between the theorical and the predicted energy")
            plt.legend()
            plt.show()

        print("The differences between the theorical values and the ones predicted by the mean cubic spline are :")
        for i in range(len(peak_e)):
            print(f"The difference for the point {i} is : {peak_e[i]-cs.mean(mu_adc[i])} keV")


        # Visualizing the difference between the predicted
        mean_delta = y_mean-y_gain
        stdmin_delta = y_mean-y_std-y_gain
        stdmax_delta = y_mean+y_std-y_gain

        if do_plot:
            plt.figure(figsize=figsize)
            plt.plot(x_gain, mean_delta, label="Mean spline - Gain spline")
            plt.plot(x_gain, stdmin_delta, label="Uncertainty min - Gain spline")
            plt.plot(x_gain, stdmax_delta, label="Uncertainty max - Gain spline")
            plt.axhline(0, linestyle="--", color ="red",  alpha=0.3)
            plt.title("Difference between the different predicted values")
            plt.xlabel("ADC values")
            plt.ylabel("Energy delta (keV)")
            plt.xlim(mu_adc[0], mu_adc[-1])
            plt.ylim(-0.025, 0.025)
            plt.legend()
            plt.show()

        
        # # Testing the linearity of the calibration
        # lin = LinearCalibration(mu_adc, peak_e, x_err=sigma_adc)
        # y_lin = lin(x_gain)
        # plt.figure(figsize=(13,9))
        # plt.plot(mu_adc, peak_e-cs(mu_adc), "bo-", label="Theorical - gain")
        # plt.plot(mu_adc, peak_e-cs.mean(mu_adc), "ro-", label="Theorical - mean")
        # plt.plot(mu_adc, peak_e-lin(mu_adc), "go-", label ="Theorical - linear")
        # plt.fill_between(mu_adc,peak_e-lower_range, peak_e-upper_range, alpha=0.3, label='Uncertainty (1σ)')
        # plt.xlabel("ADC Value")
        # plt.ylabel("Theorical - Predicted Energy (keV)")
        # plt.title("Difference between the theorical and the predicted energy")
        # plt.legend()
        # plt.show()

        # Visualizing the adc value against the uncertainties
        if do_plot:
            plt.figure(figsize=(13,9))
            plt.plot(mu_adc, mu_adc_sig, "bo")
            # plt.legend()
            plt.xlabel("ADC Value")
            plt.ylabel("Delta ADC")
            plt.title("ADC value vs uncertainty")
            plt.show()

        
        # Filtering the peaks with too large of an uncertainty
        e_sig = cs.uncertainty(mu_adc)
        mask_sig = e_sig < 0.0020 # keV or 2 eV
        mu_adc_filt = mu_adc[mask_sig]
        peak_e_filt = peak_e[mask_sig]
        mu_adc_sig_filt = mu_adc_sig[mask_sig]

        cs_filt = CubicSplineWithUncertainty(mu_adc_filt, peak_e_filt, x_err=mu_adc_sig_filt)
        
        y_gain_filt = cs_filt(x_gain)
        y_mean_filt = cs_filt.mean(x_gain)
        y_std_filt = cs.uncertainty(x_gain)

        # Visualizing the predicted value and mean against the uncertainties
        if not filter_uncertainty:
            if do_plot:
                plt.figure(figsize=figsize)
                plt.plot(cs(mu_adc), e_sig, "bo", label="Cubic spline")
                plt.plot(cs.mean(mu_adc), e_sig, "ro", label="Cubic spline mean")
                plt.legend()
                plt.xlabel("Energy (keV)")
                plt.ylabel("Delta E (keV)")
                plt.title("Predicted value and mean vs uncertainty")
                plt.show()

        # Calibrating on the filtered peaks
        else :    
            # Visualizing the gain curve
            plt.figure(figsize=figsize)
            plt.errorbar(mu_adc[~mask_sig], peak_e[~mask_sig],xerr=mu_adc_sig[~mask_sig], fmt="rx", label="Filtered out points")
            plt.plot(x_gain, y_gain, label = "Gain spline Unfiltered")
            plt.plot(x_gain, y_mean, "y--", label="Mean spline Unfiiltered")
            plt.fill_between(x_gain, y_mean - y_std, y_mean + y_std, alpha=0.3, label='Uncertainty (1σ) Unfiltered')
            plt.errorbar(mu_adc[mask_sig], peak_e[mask_sig],xerr=mu_adc_sig[mask_sig], fmt="bx", label="Filtered in points")
            plt.plot(x_gain, y_gain_filt, label = "Gain spline Filtered")
            plt.plot(x_gain, y_mean_filt, "g--", label="Mean spline Filtered")
            plt.fill_between(x_gain, y_mean_filt - y_std_filt, y_mean_filt + y_std_filt, alpha=0.3, label='Uncertainty (1σ) Filtered')
            
            plt.legend()
            plt.xlabel('ADC Values')
            plt.ylabel('Energy values (keV)')
            plt.title(f'Gain curve of Channel {ch}')
            plt.show()

        # Calculating the energy of the spectrum
        if not filter_uncertainty:
            mask_adc = (ds.p_filt_value_dc[:] > mu_adc[0] - 100) & (ds.p_filt_value_dc[:] < mu_adc[-1] + 100)
            mask_adc = ~mask_adc

            print(f"The first ADC value considered is {mu_adc[0]:.2f} and the last one is {mu_adc[-1]:.2f}")

            if np.any(mask_adc):
                print("At least one element is selected by the mask.")
            else:   
                print("No elements are selected by the mask.")  
            print("Number of elements selected:", np.sum(mask_adc))

            ds.p_energy[:] = cs(ds.p_filt_value_dc[:])
            e_mean = cs.mean(ds.p_filt_value_dc[:])
            e_std = cs.uncertainty(ds.p_filt_value_dc[:])
            ds.p_energy[mask_adc] = None
            e_mean[mask_adc] = None
            e_std[mask_adc] = None

            is_all_none = np.all(ds.p_energy[:]==None)
            print(f"The energy array is full of None values : {is_all_none}")
            print(f"The energy array has nan values : {np.isnan(ds.p_energy[:]).any()}")
            if is_all_none:
                raise ValueError("The channel doesn't have any energy values")
            print(f"\n The channel is calibrated between {min(ds.p_energy[:]):.3f} and {max(ds.p_energy[:]):.3f} keV")

            # Plotting
            c_e, b_e = np.histogram(ds.p_energy[:], bins=6000, range=(np.nanmin(ds.p_energy[:]), np.nanmax(ds.p_energy[:])))
            cg_e, bg_e = np.histogram(ds.p_energy[g], bins=6000, range=(np.nanmin(ds.p_energy[:]), np.nanmax(ds.p_energy[:])))
            plt.figure(figsize=figsize)
            plt.plot(b_e[:-1], c_e, label="all")
            plt.plot(bg_e[:-1], cg_e, label="good")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.title(f'Channel {ch} Energy Spectrum (Cubic Spline Calibration)')
            plt.legend()
            plt.yscale("log")
            plt.show()

        else :
            mask_adc = (ds.p_filt_value_dc[:] > mu_adc[mask_sig][0] - 100) & (ds.p_filt_value_dc[:] < mu_adc[mask_sig][-1] + 100)
            mask_adc = ~mask_adc
            print(f"The first ADC value considered is {mu_adc[mask_sig][0]:.2f} and the last one is {mu_adc[mask_sig][-1]:.2f}")

            if np.any(mask_adc):
                print("At least one element is selected by the mask.")
            else:   
                print("No elements are selected by the mask.")  
            print("Number of elements selected:", np.sum(mask_adc))

            ds.p_energy[:] = cs_filt(ds.p_filt_value_dc[:])
            e_mean = cs_filt.mean(ds.p_filt_value_dc[:])
            e_std = cs_filt.uncertainty(ds.p_filt_value_dc[:])

            ds.p_energy[mask_adc] = None
            e_mean[mask_adc] = None
            e_std[mask_adc] = None

            is_all_none = np.all(ds.p_energy[:]==None)
            print(f"The energy array is full of None values : {is_all_none}")
            print(f"The energy array has nan values : {np.isnan(ds.p_energy[:]).any()}")
            if is_all_none:
                raise ValueError("The channel doesn't have any energy values")
            print(f"\n The channel is calibrated between {min(ds.p_energy[:]):.3f} and {max(ds.p_energy[:]):.3f} keV")

            c_e, b_e = np.histogram(ds.p_energy[:], bins=6000, range=(np.nanmin(ds.p_energy[:]), np.nanmax(ds.p_energy[:])))
            cg_e, bg_e = np.histogram(ds.p_energy[g], bins=6000, range=(np.nanmin(ds.p_energy[:]), np.nanmax(ds.p_energy[:])))
            plt.figure(figsize=figsize)
            plt.plot(b_e[:-1], c_e, label="all")
            plt.plot(bg_e[:-1], cg_e, label="good")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.title(f'Channel {ch} Energy Spectrum (Cubic Spline Calibration)')
            plt.legend()
            plt.yscale("log")
            plt.show()
        

        if (np.nanmin(ds.p_energy[:]) < peak_e[0]-10) or (np.nanmax(ds.p_energy[:]) > peak_e[-1] +10 ):
            raise ValueError("The calibration went out of bounds")
        
        output = {
            "cs" : cs,
            "cs_filt" : cs_filt,
            "mu_adc" : mu_adc,
            "mu_adc_sig" : mu_adc_sig,
            "e_sig" : e_sig,
            "e_mean" : e_mean,
            "e_std" : e_std,
            "mask_sig" : mask_sig,
            "hist_e" : [cg_e, bg_e],
        }
        return output
    

def study_cubicspline_leave_one_out(ch, channel, peak_val, plot=True, bc1_type='not-a-knot', bc2_type='not-a-knot'):
    """
    For a given channel, fit a CubicSpline to all peaks, then refit leaving out each peak in turn.
    Plots and prints the effect of leaving out each peak.

    Parameters:
    - ch: channel number (int)
    - peak_data: list of dicts [calp_dict, count_dict, adc_val_dict]
    - peak_val: list of reference energies (keV)
    - plot: if True, show plots
    - bc_type: boundary condition for CubicSpline
    """
    adc_points = channel[ch]['cal_peaks']
    energy_points = np.array(peak_val)
    n = len(adc_points)

    # Fit with all points
    cs_full = CubicSpline(adc_points, energy_points, bc_type=bc1_type)
    x_fit = np.linspace(min(adc_points)-300, max(adc_points)+300, 500)
    y_fit_full = cs_full(x_fit)

    print(f"Study for channel {ch} :")
    for i in range(n):
        # Leave out the i-th point
        adc_leave = np.delete(adc_points, i)
        energy_leave = np.delete(energy_points, i)
        cs_leave = CubicSpline(adc_leave, energy_leave, bc_type=bc2_type)
        y_fit_leave = cs_leave(x_fit)

        if plot:
            plt.figure(figsize=(5, 3.5))
            plt.plot(adc_points, energy_points, 'ko', label='All points')
            plt.plot(adc_points[i], energy_points[i], 'ro', label=f'Removed point {i+1}')
            plt.plot(x_fit, y_fit_full, 'b-', label='CubicSpline (all)')
            plt.plot(x_fit, y_fit_leave, 'g--', label='CubicSpline (leave-one-out)')
            plt.xlabel('ADC Value')
            plt.ylabel('Energy (keV)')
            plt.title(f'Channel {ch} CubicSpline Leave-One-Out (removed {i+1})')
            plt.legend()
            plt.tight_layout()
            plt.show()

    for i in range(n):
        # Leave out the i-th point
        adc_leave = np.delete(adc_points, i)
        energy_leave = np.delete(energy_points, i)
        cs_leave = CubicSpline(adc_leave, energy_leave, bc_type=bc2_type)
        y_fit_leave = cs_leave(x_fit)
        left_distance, right_distance = horizontal_distances_to_neighbors(adc_points, i)
        # Print the effect at the removed point
        pred = cs_leave(adc_points[i])
        print(f"Removed peak {i+1}: ADC={adc_points[i]:.1f}, True={energy_points[i]:.3f} keV, Predicted={pred:.3f} keV, Error={abs(pred-energy_points[i]):.3f} keV, Left distance={left_distance:.2f} ADC, Right distance={right_distance:.2f} ADC")
    print("\n")


# def study_cubicsplinewithuncertainty_leave_one_out(calib_output, peak_e, do_plot=True, verbose=True):
#     """
#     Leave-one-out study for CubicSplineWithUncertainty.
#     For each calibration point, fit a spline leaving it out, predict its energy,
#     and check if the true value is within the 1σ error band.
#     Optionally plots the results.

#     Parameters:
#     - calib_output: dict, output from calibration()
#     - peak_e: array-like, true energy values for each peak
#     - do_plot: bool, plot the results
#     - verbose: bool, print details
#     """
#     mu_adc = np.array(calib_output["mu_adc"])
#     mu_adc_sig = np.array(calib_output["mu_adc_sig"])
#     n = len(mu_adc)

#     pred_list = []
#     pred_mean_list = []
#     pred_std_list = []
#     within_list = []
#     within2_list = []
#     error_std_list = []

#     for i in range(n):
#         mask = np.ones(n, dtype=bool)
#         mask[i] = False
#         cs_loo = CubicSplineWithUncertainty(mu_adc[mask], peak_e[mask], x_err=mu_adc_sig[mask])
#         pred = cs_loo(mu_adc[i])
#         pred_mean = cs_loo.mean(mu_adc[i])
#         pred_std = cs_loo.uncertainty(mu_adc[i])
#         theor = peak_e[i]
#         within = abs(theor - pred_mean) <= pred_std
#         within2 = abs(theor - pred_mean) <= 2*pred_std
#         # error = abs(theor-pred)
#         error_std = theor - pred_mean
#         pred_list.append(pred)
#         pred_mean_list.append(pred_mean)
#         pred_std_list.append(pred_std)
#         within_list.append(within)
#         within2_list.append(within2)
#         error_std_list.append(error_std)
#         if verbose:
#             print(f"Removed peak {i}: ADC={mu_adc[i]:.2f}, True={theor:.4f} keV, "
#                 #   f"Predicted (without uncertainty) ={pred:.4f} keV, "
#                   f"Predicted ={pred_mean:.4f} ± {pred_std:.4f} keV, "
#                   f"Within 1σ: {'YES' if within else 'NO'}, "
#                   f"Within 2σ: {'YES' if within2 else 'NO'}, "
#                 #   f"Error (without uncertainty) = {error:.4f} keV, "
#                   f"Error = {error_std:.4f} keV")

#     if do_plot:
#         plt.figure(figsize=(8,5))
#         plt.errorbar(mu_adc, pred_mean_list, yerr=pred_std_list, fmt='o', label='Predicted (with uncertainty)')
#         # plt.plot(mu_adc, pred_list, "bx", label="Predicted (without uncertainty)")
#         plt.plot(mu_adc, peak_e, 'rx', label='True value')
#         for i, ok in enumerate(within_list):
#             if not ok:
#                 plt.plot(mu_adc[i], pred_mean_list[i], 'ko', markerfacecolor='none', markersize=12, label='Outside 1σ' if i==within_list.index(False) else "")
#         for i, ok in enumerate(within2_list):
#             if not ok:
#                 plt.plot(mu_adc[i], pred_mean_list[i], 'ro', markerfacecolor='none', markersize=15, label='Outside 2σ' if i==within2_list.index(False) else "")
#         plt.xlabel('ADC Value')
#         plt.ylabel('Energy (keV)')
#         plt.title('Leave-one-out CubicSplineWithUncertainty\nPrediction vs True Value')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#     return pred_mean_list, pred_std_list, within_list, within2_list, error_std_list

def study_cubicsplinewithuncertainty_leave_one_out(
    calib_output, peak_e, do_plot=True, verbose=True, use_mask=False
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
    mu_adc = np.array(calib_output["mu_adc"])
    mu_adc_sig = np.array(calib_output["mu_adc_sig"])
    n = len(mu_adc)

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

    for i in range(n):
        mask = used_mask.copy()
        if i in loo_indices:
            mask[i] = False  # leave this one out
            cs_loo = CubicSplineWithUncertainty(mu_adc[mask], peak_e[mask], x_err=mu_adc_sig[mask])
            pred = cs_loo(mu_adc[i])
            pred_mean = cs_loo.mean(mu_adc[i])
            pred_std = cs_loo.uncertainty(mu_adc[i])
            theor = peak_e[i]
            within = abs(theor - pred_mean) <= pred_std
            within2 = abs(theor - pred_mean) <= 2*pred_std
            error_std = theor - pred_mean
            if verbose:
                print(f"Removed peak {i}: ADC={mu_adc[i]:.2f}, True={theor:.4f} keV, "
                    f"Predicted ={pred_mean:.4f} ± {pred_std:.4f} keV, "
                    f"Within 1σ: {'YES' if within else 'NO'}, "
                    f"Within 2σ: {'YES' if within2 else 'NO'}, "
                    f"Error = {error_std:.4f} keV")
        else : 
            cs_loo = CubicSplineWithUncertainty(mu_adc[mask], peak_e[mask], x_err=mu_adc_sig[mask])
            pred = cs_loo(mu_adc[i])
            pred_mean = cs_loo.mean(mu_adc[i])
            pred_std = cs_loo.uncertainty(mu_adc[i])
            theor = peak_e[i]
            within = abs(theor - pred_mean) <= pred_std
            within2 = abs(theor - pred_mean) <= 2*pred_std
            error_std = theor - pred_mean
            if verbose:
                print(f"Filtered peak {i}: ADC={mu_adc[i]:.2f}, True={theor:.4f} keV, "
                    f"Predicted ={pred_mean:.4f} ± {pred_std:.4f} keV, "
                    f"Within 1σ: {'YES' if within else 'NO'}, "
                    f"Within 2σ: {'YES' if within2 else 'NO'}, "
                    f"Error = {error_std:.4f} keV")
                
        pred_list.append(pred)
        pred_mean_list.append(pred_mean)
        pred_std_list.append(pred_std)
        within_list.append(within)
        within2_list.append(within2)
        error_std_list.append(error_std)
        

    if do_plot and len(loo_indices) > 0:
        plt.figure(figsize=(8,5))
        plt.errorbar(mu_adc, pred_mean_list, yerr=pred_std_list, fmt='o', label='Predicted (with uncertainty)')
        plt.plot(mu_adc, np.array(peak_e), 'rx', label='True value')
        for i, ok in enumerate(within_list):
            if not ok:
                plt.plot(mu_adc[i], pred_mean_list[i], 'ko', markerfacecolor='none', markersize=12, label='Outside 1σ' if i==within_list.index(False) else "")
        for i, ok in enumerate(within2_list):
            if not ok:
                plt.plot(mu_adc[i], pred_mean_list[i], 'ro', markerfacecolor='none', markersize=15, label='Outside 2σ' if i==within2_list.index(False) else "")
        plt.xlabel('ADC Value')
        plt.ylabel('Energy (keV)')
        plt.title('Leave-one-out CubicSplineWithUncertainty\nPrediction vs True Value (used points)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return pred_mean_list, pred_std_list, within_list, within2_list, error_std_list

def coadd_energy_histograms(data, chans, energy_range=(20, 120), nbins_e=6000, do_plot=True):
    """
    Coadd energy histograms for a list of channels.

    Parameters:
    - data: MASS TESGroup object
    - chans: list of channel numbers
    - energy_range: tuple, (min, max) energy in keV
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
        plt.xlabel('energy (keV)')
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
    plt.xlabel('energy (keV)')
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
#         plt.xlabel("Energy (keV)")
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
        plt.xlabel("Energy (keV)")
        plt.title("Kernel Density Estimation of the Energy")
        plt.show()
    return total