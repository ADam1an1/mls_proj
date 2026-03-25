# plot_func.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logarithmic(a, b, x):
    return a * np.log(x) + b

def linear(m, b, x):
    return m * x + b

def parabolic(a, b, c, x):
    return a * ((x - b) ** 2) + c


def plot_all(result, system_e_levels, freqs, time, steps, operators, save=False, save_name=""):
    """
    Plots energy, photon number, and system state operators
    Saves plot in specified location 
    """
    tlist = np.linspace(0, time, steps)

    shift = 0
    if "energy" in operators:
        energy = result.expect[0]
        plt.figure(figsize=(12,6))
        plt.plot(tlist, energy,
                         label="Total System Energy", ls="-")
        plt.xlabel("Time (a.u.)")
        plt.legend(loc="upper right")
        plt.title("Energy")
        plt.show()

        shift += 1

        if save:
            try:
                plt.savefig(save_name + "_energy.png")
            except:
                print("No image file name given")

    if "photons" in operators:
        plt.figure(figsize=(12,6))
        for alpha in range(len(freqs)):
            photon = result.expect[alpha + shift]
            plt.plot(tlist, photon,
                             label="Photon Number, Freq = {}".format(freqs[alpha]),
                             alpha=0.5, marker='|', markersize=0, ls="-")
        plt.xlabel("Time (a.u.)")
        plt.legend(loc="upper right")
        plt.title("Photon Number")
        plt.show()

        shift += len(freqs)

        if save:
            try:
                plt.savefig(save_name + "_photonnumbers.png")
            except:
                print("No image file name given")

    if "states" in operators:
        for sys in range(len(system_e_levels)):
            plt.figure(figsize=(12,6))
            for i in range(len(system_e_levels[sys])):
                state = result.expect[shift + i]
                plt.plot(tlist, state,
                                 label="Energy = {}".format(system_e_levels[sys][i]),
                                 alpha=0.5, linewidth=1.5)
            shift += len(system_e_levels[sys])

            plt.xlabel("Time (a.u.)")
            plt.legend(loc="upper right")
            plt.title("State for System {}".format(sys))
            plt.show()


            if save:
                try:
                    plt.savefig(save_name + "_sys{}states.png".format(sys))
                except:
                    print("No image file name given")




def plot_multi_fourier(indices, result, time, steps, positionings, photons=[], power=False):
    plt.figure(figsize=(12, 6))
    for i in range(len(indices)):
        fft = np.fft.fft(result, expect[index])
        freqs = np.fft.fftfreq(steps, time / steps)
        
        if len(photons) and i < len(photons):
            label = "Photon Freq: {}".format(photons[i])
        else:
            label = "System: {}".format(positionings[i])
            
        if power:
            plt.title("Power Spectrum for Varying Positionings")
            plt.plot(freqs[:int(len(freqs)/2)], np.abs(fft**2)[:int(len(freqs)/2)], label=label)
    else:
            plt.title("Fourier Transform")
            plt.plot(freqs, fft, label=labels[index])
    plt.xlabel("Frequency")
    plt.legend()
    plt.show()


def plot_fourier(indices, result, time, steps, power=False, labels=[]):
    """
    Plots fourier or power spectra for indices specified together
    """
    plt.figure(figsize=(12,6))
    for index in indices:
        fft = np.fft.fft(result.expect[index])
        freqs = np.fft.fftfreq(steps, time / steps)
        
        if not power:
            plt.title("Fourier Transform")
            plt.plot(freqs, fft, label=labels[index])
        else:
            plt.title("Power Spectrum")
            plt.plot(freqs[:int(len(freqs)/2)], np.abs(fft**2)[:int(len(freqs)/2)], label=labels[index])
    
    plt.xlabel("Frequency")
    plt.legend()
    plt.show()

    if save:
        try:
            plt.savefig(save_name + "_fourier{}.png".format(str(indices)))
        except:
            print("No image file name given")

