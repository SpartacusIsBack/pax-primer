# am241lib — TES Microcalorimeter Calibration Library

This library provides tools to calibrate a TES (Transition Edge Sensor) microcalorimeter detector using measured spectra. It performs peak finding, fitting (with Gaussians and splines), and calibration curve construction, and includes utilities for visualization, uncertainty estimation, and validation.

---

## Table of Contents
- [Overview](#overview)
- [Functions](#functions)
- [Classes](#classes)
- [Calibration Workflow](#calibration-workflow)

---

## Overview

The library supports:
- Gaussian and double Gaussian fitting to peaks in histograms.
- Calibration curves using cubic splines with uncertainty bands.
- Visualization of spectra, fitted peaks, and calibration curves.
- Leave-One-Out (LOO) validation of calibration points.
- Full-channel calibration and coadding results from multiple channels.

---

## Functions

### `gaussian(x, a, mu, sigma, c)`
Gaussian function with amplitude `a`, center `mu`, standard deviation 
`sigma`, and constant background `c`.  
  
**Parameters:**
- `x`: array-like, independent variable (PH or Energy).
- `a`: float, amplitude of peak.
- `mu`: float, center of peak.
- `sigma`: float, standard deviation.
- `c`: float, constant offset.

### `weighted_mse(errors, uncertainties, use_variance=False)`
Computes the weighted mean squared error (MSE) between errors and uncertainties.  
If `use_variance` is True, weights by the inverse variance; otherwise, by the inverse uncertainty.
  
**Parameters:**
- `errors`: array-like, error values.
- `uncertainties`: array-like, uncertainty values.
- `use_variance`: bool, use variance for weighting if True.

### `fit_fwhm(x, y, plot=True, in_energy=False, return_sig=False)`
Fit a Gaussian to a spectrum and return FWHM and optimal parameters.  
  
**Parameters:**
- `x`: array-like, bin centers.
- `y`: array-like, counts.
- `plot`: bool, whether to display the fit.
- `in_energy`: bool, label axes in energy units if True.
- `return_sig`: bool, whether to also return uncertainties.


### `double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2, c)`
Sum of two Gaussian peaks and a constant background.
  
**Parameters:**
- Same as `gaussian`, but for two peaks (`a1`, `mu1`, `sigma1`, `a2`, `mu2`, `sigma2`) and constant `c`.

### `fit_double_gaussian(x, y, plot=True, in_energy=False, return_sig=False)`
Fit two Gaussian peaks to a spectrum, returning FWHMs and parameters.
  
**Parameters:**
- `x`: array-like, bin centers.
- `y`: array-like, counts.
- `plot`: bool, whether to display the fit.
- `in_energy`: bool, label axes in energy units if True.
- `return_sig`: bool, whether to also return uncertainties.


### `linear_fit(p, x)`
Simple linear function `a*x + b`.
  
**Parameters:**
- `p`: tuple/list of parameters `(a, b)`.
- `x`: array-like.


### `horizontal_distances_to_neighbors(points_x, idx)`
Compute distances to neighboring calibration points in Pulse Height.
  
**Parameters:**
- `points_x`: array-like, PH calibration points.
- `idx`: int, index of the point to check.


### `data_treat(run_pulse, run_noise, files_to_delete)`
Load and preprocess TES data from files, returning a TESGROUP data object and an array containing time related data.
  
**Parameters:**
- `run_pulse`: str, pattern for pulse files.
- `run_noise`: str, pattern for noise files.
- `files_to_delete`: list of filenames to delete before processing.


### `log_find_peaks(counts, height=None, distance=None, prominence=None)`
Find peaks in log-scaled histogram data.
  
**Parameters:**
- `counts`: array-like, histogram counts.
- `height`, `distance`, `prominence`: optional peak-finding criteria.


### `add_channel_histogram_to_ax(data, ch, ax, bins=5000, hist_range=(0, 4e3), Timebase = None, do_plot = True, do_peaks = True, do_allg=False, return_val=False, height=None, distance=None, prominence=None)`
Plot the histogram of a single channel on given matplotlib axes, optionally marking peaks.
  
**Parameters:**
- `data`: TESGroup data object.
- `ch`: int, channel number.
- `ax`: matplotlib axes to plot on.
- Other parameters control bins, plotting, filtering, etc.


### `find_biggest_peak(counts, bin_edges)`
Find the biggest peak in a histogram.
  
**Parameters:**
- `counts`: histogram counts.
- `bin_edges`: edges corresponding to counts.


### `get_bin_count(ph_value, counts, bin_edges)`
Get counts in the bin corresponding to a given Pulse Height.
  
**Parameters:**
- `ph_value`: float, PH to query.
- `counts`, `bin_edges`: histogram.


### `zoom_in_on_peaks(ds, peak_data, peak_index, zoom_factor=0.1, zbins=200, hheight=True, do_plot=True)`
Zoom into a specific peak in Pulse Height and display it alongside the full spectrum.
  
**Parameters:**
- `ds`: dataset for one channel.
- `peak_data`: array-like, contains the calibration peak's index, count, PH dictionaries
- `peak_index`: int, which peak to zoom on.
- `zoom_factor`: relative zoom width.
- `zbins`: histogram bins.
- `hheight`: whether to show half-height line.


### `zoom_all_e(data, ch, timebase=None, zoom_factor=0.1, zbins=200, hheight=True, return_val=False, distance = 30, prominence = 1, height = None)`
Zooms on all the peaks identified with `scipy.find_peaks` of an energy spectrum made with an approximative linear calibration
  
**Parameters:**
- `data`: dataset for one channel.
- `ch`: int, channel number
- `timebase`: array-like, time from trigger array
- `zoom_factor`: float, relative zoom width.
- `zbins`: int, histogram bins.
- `hheight`: bool, whether to show half-height line.
- `distance`: float, minimum distance between two peaks in PH, parameter of `find_peaks`.
- `prominence`: float, prominence parameter of `find_peaks`.
- `height`: float, minimum height of the peak, parameter of `find_peaks`. 

### `zoom_in_ph(ds, ph_value, zoom_factor=0.1, zbins=200, do_plot = True)`
Zoom into a region around a specified Pulse Height.
  
**Parameters:**
- `ds`: dataset for one channel.
- `adc_value`: PH to center zoom on.

### `zoom_in_energy(ds, e_value, zoom_factor=0.1, zbins=200, do_plot = True)`
Zoom into a region around a specified Energy in eV.
  
**Parameters:**
- `ds`: dataset for one channel.
- `adc_value`: PH to center zoom on.


### `Calibrate_all(data, peak_e, pk_name, e_unc=None, Timebase=None, escp_pk_e= None, do_plot=False, cal_type="MASS")`
Calibrate all channels using specified calibration peaks
  
**Parameters:**
- `data`: TESGroup data.
- `peak_e`: list of calibration energies (in eV).
- `pk_name`: string list of names of each calibration energies.
- `e_unc`: uncertainties on calibration energies (in eV).
- `Timebase`: timing data.
- `escp_pk_e`: optional escape peak energies (in eV).
- `do_plot`: bool, whether to plot all graphs of the calibration or only the essentials.
- `cal_type`: string, is either `"MASS"` or `"CubicSpline"`, chooses which method to use for the splines.


### `loo(...)`
Perform Leave-One-Out test on calibration curve (Energy domain).
  
**Parameters:**
- `calib_output`: output from calibration.
- `peak_e`: calibration energies.
- `e_unc`: uncertainties.
- `do_plot`, `verbose`, `use_mask`: options.


### `loo_gain(...)`
Perform Leave-One-Out test on gain calibration (PH/Energy).
  
**Parameters:**
- `calib_output`: output from calibration.
- `peak_e`: calibration energies.
- `e_unc`: uncertainties.
- `do_plot`, `verbose`, `use_mask`: options.


---

## Classes

### `Coadd_data`

Collects and coadds data from multiple channels. Behaves the same way as channel dataset class (can be used in place of ds for the functions of this library).

- **Constructor Parameters**:
  - `data`: TESGroup data.
  - `timebase`: timing data.
  - `calibration_output`: calibration results.
  - `ch_list`: list of channel numbers (usually the calibrated channels).
  - `peak_e =None`: array of the calibration energies
  - `peak_name =None`: list of calibration points' names
  - `escp_pk_e =None`: array of the escape peak's energies
- **Attributes**: Pulse Height, Energy, rise times, etc. from all channels.
- **Methods**:
  - `good()`: returns mask of good events.
  - `hist(...)`: histogram of coadded Energy.
  - `plot_offset_spectra(...)`: plots spectra of all channels with offsets.
  - `peaktest(zbins = 600, do_plot = False)`: tests if the known peaks of the coadd have the same values as the calibration energies and plots their FWHM.

### `CubicSplineWithUncertainty`

Cubic spline interpolator with Monte Carlo uncertainty estimation.

- **Constructor**:
  - `x`, `y`: calibration points.
  - `x_err`, `y_err`: uncertainties.
  - `n_samples`: number of Monte Carlo samples.
- **Methods**:
  - `__call__(x_eval)`: evaluate spline at `x_eval`.
  - `mean(x_eval)`: mean prediction.
  - `uncertainty(x_eval)`: standard deviation.
  - `gain(x_eval)`, `g_mean(x_eval)`, `g_uncertainty(x_eval)`: gain curve and uncertainty.
  - `min(x_eval, n_sigma=1)`, `max(x_eval, n_sigma=1)`: lower/upper bounds at n sigmas.

### `LinearCalibration`

Simple linear calibration using Orthogonal Distance Regression.

- **Constructor Parameters**:
  - `x`, `y`, `x_err`, `y_err`: calibration points and uncertainties.
- **Methods**:
  - `__call__(x_eval)`: evaluate line at `x_eval`.
  - `return_parameter()`: return fitted slope, intercept, and errors.

### `Calibration`

Main class to perform channel calibration.

- **Constructor**: 
  - `data`: TESGroup data.
  - `peak_e`: list of calibration energies.
  - `pk_name`: list of calibration points' names
  - `e_unc`: uncertainties on calibration energies.
  - `Timebase`: timing data.
  - `escp_pk_e`: optional escape peak energies.

- **Methods**:
  - `__call__(ch, zoom_bins = 700, do_plot = True, cal_type="MASS")`: calibrates the given channel using either the `"MASS"` or `"CubicSpline"` method.


### `LooTest`

Performs a Leave-One-Out (LOO) validation for TES channel calibrations.  
For each calibrated channel, the class computes the error between the predicted and true calibration peak energies (or gain) when each calibration point is left out in turn.  
Provides a method to visualize these errors for all channels.

- **Constructor Parameters**:
  - `calibrated_ch`: list of calibrated channel numbers.
  - `output`: dict of calibration outputs per channel (from `Calibrate_all`).
  - `peak_e`: array of calibration energies (eV).
  - `e_unc`: array of calibration energy uncertainties (eV).
  - `cal_space`: `"energy"` (default) or `"gain"`, selects the test domain.

- **Attributes**:
  - `results`: dict keyed by channel, each containing arrays of predicted means, uncertainties, and error statistics for each calibration peak.

- **Methods**:
  - `plot(max_per_fig=7, ylim=None, use_idx=False)`:  
    Plots the difference between predicted and true values for each peak, for all channels.  
    - `max_per_fig`: int, maximum number of channels per figure.
    - `ylim`: tuple, y-axis limits.
    - `use_idx`: bool, if True, x-axis is peak index; if False, x-axis is peak energy.

---

## Calibration Workflow
0. **Import the library**
   ```python
   import TEScal as tes
   ```

1. **Data preparation:**
   Identify the path to the data pulse and noise as well as the hdf5 files from past MASS treatments
   ```python
   data, timebase = tes.data_treat("pulse_path", "noise_path", files_to_delete=["pulse_file.hdf5", "noise_file.hdf5"])
   ```

2. **Visualize the data**
   - Check to see if the data has been treated correctly :
   ```python
   print("Good channels:", data.good_channels)
   print("All channels:", list(data.channel.keys()))
   ```
   - Plot the Pulse Height spectrum of all the good channels :
   ```python
   peaks_dict = {}
   adc_val_dict = {}
   count_dict = {}

   height = None
   distance = 30
   prominence = 1
   for ch in data.good_channels:
      plt.figure(figsize=(5,3.5))
      counts, bins, peaks_dict[ch], _ = tes.add_channel_histogram_to_ax(data, ch, plt, Timebase=timebase, do_peaks= True, distance=distance, prominence=prominence, return_val=True)
      plt.show()
   ```
   The `height`, `distance`, `prominence` are parameters that can be tweaked to allow more or less peaks to be marked.
   (The `peaks_dict`, `adc_val_dict`, `count_dict` are used to create the `peak_data` array later on)

   - Choose a channel that you deem good and zoom in on its individual peaks :
   ```python
    ch = ...
    tes.zoom_all_e(data, ch, timebase)
   ```
   

3. **Peak selection:**
   Identify the known calibration peaks.
   Define calibration energies (`peak_e`) and uncertainties (`e_unc`) as arrays.
   If there are any escapes peaks or k-alpha peaks, note their energies inside the `peak_e` array as well as the `esc_pk_e` array.
   Note that both the `e_unc` and `esc_pk_e` arrays are optional and not required for the `Calibrate_all` function and `Calibration` class to work.

4. **Calibration:**
   Pass the necessary parameters to the `Calibrate_all` function.
   ```python
   cal, coadded_data, outputs, calibrated_ch, failed_ch, error_log = Calibrate_all(
       data, peak_e, e_unc=e_unc, Timebase=timebase, escp_pk_e=None, do_plot=True
   )
   ```
   It will return the following objects :
   - `cal`: [`Calibration`](#calibration) class object, A tool that can be used to calibrate individually each channel.
   - `coadded_data`: [`Coadd_data`](#coadd_data) class object, An object that can be used the same way as a channel dataset with functions of this library.
   - `outputs`: dict, Dictionary of the outputs of `Calibration` for each channel. Its keys are the channel numbers.
   - `calibrated_ch`: list, List of all the channels that were calibrated successfully.
   - `failed_ch`: list, List of all the channels that failed to be calibrated.
   - `error_log`: dict, Dictionary of all the errors that caused the failed calibrations. Its keys are the channel numbers.

5. **LOO validation and Coadd Visualisation (optional):**
   You can perform a leave-one-out test to see if the calibrations are valid or not.
   You can choose the calibration test space by giving either the `"energy"` or `"gain"` string to the `cal_space` parameter.
   ```python
   loo = tes.LooTest(calibrated_ch, output, peak_e, e_unc, cal_space="energy")
   loo.plot()
   ```
   To visualize the coadd data you can simply use the `.hist()` method of the [`Coadd_data`](#coadd_data) class.
   ```python
   coadded_data.hist()
   ```
   You can also perform a test to check if the coadd's peaks are aligned with the calibration energies and measure their FWHM
   ```python
   coadded_data.peaktest()
   ```

---

## Notes
- All measured detector signals are treated as **Pulse Height**.
- The calibration assumes approximate linearity between PH and energy, as well as the tallest peak being that of the ~59 keV 241am to find the peaks nearby the given energies. If the "main" peak is not at ~59 keV, correct the value of the `main_e` variable at the beginning of the TESCal.py file (in eV).
- The library uses `matplotlib`, `numpy`, `scipy`, `pandas`, and the MASS TES analysis framework.
- Various plots can be optionally enabled to visualize calibration and spectra.

---

## Authors
Library developed for calibrating TES microcalorimeter detectors using Am-241 and other calibration sources by Alexandre Le Dréo.

---

For detailed examples or help, please refer to the source code or contact the maintainer at the email : alexandre.le_dreo@etu.sorbonne-universite.fr

