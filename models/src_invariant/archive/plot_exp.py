# plot_exp.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logarithmic(x, a, h, k):
    return a * np.log(x - h) + k

def linear(x, m, b):
    return m * x + b

def parabolic(a, b, c, x):
    return a * ((x - b) ** 2) + c

def plot_avg_pn(photon_nums, system_nums, 
        plt_title="Average Photon Number", xlabel="N Systems",
        ylabel="",
        norm=False):
    
    if norm:
        scaling = np.mean(photon_nums[0])
    else:
        scaling = 1

    plt.figure(figsize=(12, 6))
    for i in range(len(system_nums)):
        plt.scatter(system_nums[i], np.mean(photon_nums[i]) / scaling)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



def plot_multi_photon_number(photon_numbers, system_nums, time, steps,
        plt_title="", leg_title="", labels=[]):
    """
    Plots the photon numbers for a set of experiments
    """


    tlist = np.linspace(0, time, steps)

    plt.figure(figsize=(12,6))
    for i in range(len(photon_numbers)):
        plt.plot(tlist, photon_numbers[i], label=labels[i], alpha=0.75)

    plt.xlabel("Time (a.u.)")
    plt.ylabel("Photon Number")
    plt.legend(loc="upper right", title=leg_title, ncols=np.ceil(len(photon_numbers)/5))
    plt.title(plt_title)
    plt.show()


def plot_fourier_multi_photon_number(photon_numbers, system_nums, time, steps,
        xrange=(0, 1), yrange=(0, 1e9 * 0.15), power=False,
        plt_title="", leg_title="", labels=[]):
    """
    Plots the fourier transform or power spectra of the photon number
    for a set of experiments
    """


    plt.figure(figsize=(12, 6))
    freqs = np.fft.fftfreq(steps, time/steps)

    for i in range(len(photon_numbers)):
        fft = np.fft.fft(photon_numbers[i])
        if power:
                plt.title(plt_title)
                plt.plot(freqs[:int(len(freqs)/2)], np.abs(fft**2)[:int(len(freqs)/2)], label=labels[i])
        else:
            plt.plot(freqs, fft, label=labels[i])

    plt.title(plt_title)
    plt.xlabel("Frequency")
    plt.xlim(xrange)
    plt.ylabel("Signal")
    plt.ylim(yrange)
    plt.legend(title=leg_title, ncols=np.ceil(len(photon_numbers)/5))
    plt.show()


def plot_signal_peaks_photon_number(photon_numbers, system_nums,
        time, steps, xrange, yrange, fit_func=None,
        plt_title="", leg_title="", x_label=""):
    """
    Plots the peaks of the fourier transform for a set of
    experiments and finds a fit for the curve
    """

    plt.figure(figsize=(12, 6))

    nyquist = time / steps / 2 
    freqs = np.fft.fftfreq(steps, time/steps)
    pos_freqs = freqs[:int(len(freqs)/2)]
    cutoff = np.argmax(np.fft.fftfreq(steps, time/steps) > nyquist)
    above_nyq_freqs = pos_freqs[cutoff:]

    peak_freqs = []
    for i in range(len(photon_numbers)):
        fft = np.fft.fft(photon_numbers[i])
        power = np.abs(fft**2)[:int(len(freqs)/2)][cutoff:]

        peak_index = np.argmax(power)
        plt.scatter(system_nums[i], above_nyq_freqs[peak_index])
        peak_freqs.append(above_nyq_freqs[peak_index])

    if fit_func == "linear":
        popt, pcov = curve_fit(linear, system_nums, peak_freqs)
    elif fit_func == "log":
        popt, pcov = curve_fit(logarithmic, system_nums, peak_freqs)
    elif fit_func == "parabolic":
        popt, pcov = curve_fit(parabolic, system_nums, peak_freqs)

    yfit = []
    for i in range(len(system_nums)):
        if fit_func == "linear":
            yfit.append(linear(system_nums[i], *popt))
        elif fit_func == "log":
            yfit.append(logarithmic(system_nums[i], *popt))
        elif fit_func == "parabolic":
            yfit.append(parabolic(popt[2], popt[1], popt[0], system_nums[i]))

    if fit_func:
        plt.plot(system_nums, yfit, "--",  label="fit", alpha = 0.5, c='k')

    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.ylim(yrange)
    plt.title(plt_title)
    plt.legend(title=leg_title)
    plt.show()



