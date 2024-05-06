import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Given parameters
N = 4
Fc = 1e9  # Carrier frequency in Hz
Fs = 1e10  # Sampling frequency in Hz
total_sampling_time = 2e-5 # Total sampling time in seconds
Fm = 1/(2e-5)  # Modulation frequency in Hz
D_lambda = 0.5  # D/lambda
total_sampling_points = 200000  # Total number of sampling points
Tp = 1e-6  # Period of the modulation function
SNR = 10 #Signal-to-Noise ratio

def pandq(theta, phi, D_lambda):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    u = np.pi * D_lambda * np.sin(theta) * np.cos(phi)
    v = np.pi * D_lambda * np.sin(theta) * np.sin(phi)
    p = (2 / np.pi) * np.tan((u + v) / 2)
    q = (2 / np.pi) * np.tan((u - v) / 2)
    mmm = p +1j*q
    #print("The theoritical ratio between b1 and b0:{0}".format(mmm))
    return(mmm)

def solvep(b1, b0, Dlambda):
    mmm = b1 / b0
    #print("The ratio between b1 and b0(+1st harm):{0}".format(mmm))
    p = mmm.real
    q = mmm.imag
    tmp = ((1/(np.pi*Dlambda))**2 *
           ((np.arctan(np.pi*p/2)+np.arctan(np.pi*q/2))**2 +
            (np.arctan(np.pi*p/2)-np.arctan(np.pi*q/2))**2))
    theta_est = np.arcsin(np.sqrt(tmp))
    tmp = ((np.arctan(np.pi*p/2)-np.arctan(np.pi*q/2))/
           (np.arctan(np.pi*p/2)+np.arctan(np.pi*q/2)))
    phi_est = np.arctan(tmp)
    return theta_est, phi_est

def solven(b1, b0, Dlambda):
    mmm = b1 / b0
    #print("The ratio between b1 and b0(-1st harm):{0}".format(mmm))
    p = mmm.real
    q = mmm.imag
    tmp = ((-np.arctan(np.pi*p/2) - np.arctan(np.pi*q/2))/
           (np.arctan(np.pi*q/2) - np.arctan(np.pi*p/2)))
    phi_est = np.arctan(tmp)
    tmp = ((1/(np.pi*Dlambda))**2 *
           ((np.arctan(np.pi*q/2) - (np.arctan(np.pi*p/2)))**2 +
            (np.arctan(np.pi*q/2) + (np.arctan(np.pi*p/2)))**2))
    theta_est = np.arcsin(np.sqrt(tmp))
    return theta_est, phi_est


# Modulation function for each antenna element
def U_n(t, n, N, Tp):
    m = np.floor(t / Tp)
    return np.where(((n - 1) / N + m) * Tp < t,
                    np.where(t < (n / N + m) * Tp, 1, 0), 0)

# Phase difference calculation
def delta_phi_n(n, N, D_lambda, theta, phi):
    return (np.pi * D_lambda * np.sin(theta)
            * np.cos(2 * (n - 1)*np.pi / N - phi))

#Adding white Gaussian Noise to the received signal
def add_gaussian_noise(signal, SNR_dB):

    signal_power = np.mean(np.abs(signal) ** 2)
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise = (np.sqrt(noise_power) *
             np.random.randn(*signal.shape))
    noisy_signal = signal + noise
    return noisy_signal

# The function used for the
# visualization of the power spectrum of a signal
def spectrum(fft, freqs, range):
    power_spectrum = np.abs(fft)**2
    fund_freq = np.argmax(power_spectrum)
    #The following line should be commented
    #if the user wants to calculate the actual
    #power spectrum of the signal, and uncommented
    #if the user wants to determine the normalized signal spectrum
    #power_spectrum = 10*np.log10(power_spectrum
    #                             / power_spectrum[fund_freq])
    freqs_range = np.where(np.logical_and
                           (freqs >= range[0], freqs <= range[1]))[0]
    plt.plot(freqs[freqs_range], power_spectrum[freqs_range])
    plt.xlabel("Frequencies [GHz]")
    plt.ylabel("Normalized Power Spectrum [dB]")
    plt.title("Normalized Power Spectrum from {0}GHz to {1}GHz"
              .format(float(range[0]/1e9), float(range[1]/1e9)))
    plt.show()

#The main function for DoA estimation with the given parameters
def DoAEstimator(theta, phi, Fc, Fs, total_sampling_time,
                 total_sampling_points, N, D_lambda, Tp, SNR, harm):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    t = np.linspace(0, total_sampling_time, total_sampling_points)
    received_signals = []
    for n in range(1, N+1):  # For each antenna element
        phase_difference = delta_phi_n(n, N, D_lambda, theta, phi)
        signal = np.exp(1j * (2 * np.pi * Fc
                            * t - phase_difference)) * U_n(t, n, N, Tp)
        received_signals.append(signal)
    total_signal = received_signals[0]
    for i in range(1,N):
        total_signal = total_signal + received_signals[i]
    total_signal = add_gaussian_noise(total_signal, SNR)
    fft_result_total_signal = np.fft.fft(total_signal, n=total_sampling_points)
    power_spectrum_total_signal = np.abs(fft_result_total_signal) ** 2
    frequency_vector = np.linspace(0, Fs, total_sampling_points)
    #spectrum(fft_result_total_signal, frequency_vector, (-5e9, 5e9))
    fundamental_freq_index = np.argmax(power_spectrum_total_signal)
    b0 = fft_result_total_signal[fundamental_freq_index]
    power_spectrum_total_signal[fundamental_freq_index] = -1
    first_harmonic_index = np.argmax(power_spectrum_total_signal)
    b1 = fft_result_total_signal[fundamental_freq_index
                                 + np.abs(fundamental_freq_index -
                                          first_harmonic_index)]
    bn1 = fft_result_total_signal[fundamental_freq_index
                                 - np.abs(fundamental_freq_index -
                                          first_harmonic_index)]
    thetaa1, phii1 = solvep(b1, b0, D_lambda)
    thetaa2, phii2 = solven(bn1, b0, D_lambda)
    return np.rad2deg(thetaa1), np.rad2deg(phii1), np.rad2deg(thetaa2), np.rad2deg(phii2)

theta = 29
phi = 50

theta1, phi1, theta2, phi2 = DoAEstimator(theta, phi, Fc, Fs, total_sampling_time,
                 total_sampling_points, N, 0.5, Tp, 10, 1)

print("Actual Theta: {0}, Actual Phi: {1}".format(theta, phi))
print("Estimated Theta: {0}, Estimated Phi: {1}".format(theta1, phi1))
print("Estimated Theta: {0}, Estimated Phi: {1}".format(theta2, phi2))
