import numpy as np
import matplotlib.pyplot as plt
import UCTMA

N = 4
Fc = 1e9
total_sampling_time = 100e-6
total_sampling_points = 500000
SNR = 0
D_lambda = 0.5

def Music_spectrum(phi_scan, theta_scan, music_spectrum_dB):
    Phi, Theta = np.meshgrid(np.degrees(phi_scan), np.degrees(theta_scan))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Phi, Theta, music_spectrum_dB, cmap='viridis')
    ax.set_xlabel('Elevation Angle (degrees)')
    ax.set_ylabel('Azimuth Angle (degrees)')
    ax.set_zlabel('MUSIC Spectrum (dB)')
    plt.colorbar(surf, ax=ax, label='MUSIC Spectrum (dB)')
    plt.title('3D MUSIC Spectrum')
    plt.show()

def delta_phi_n(n, N, D_lambda, theta, phi):
    return (np.pi * D_lambda * np.sin(theta)
            * np.cos(2 * (n - 1)*np.pi / N - phi))

def DoaMusic(theta, phi, Fc, total_sampling_points,
             total_sampling_time, N, D_lambda, SNR):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    n = np.arange(N)
    t = np.linspace(0, total_sampling_time, total_sampling_points)
    A = np.exp(1j * delta_phi_n(n, N, D_lambda, theta, phi))
    s_t = np.exp(1j * 2 * np.pi * Fc * t)
    SNR = 10 ** (SNR / 10)
    noise_power = 1 / SNR
    noise = np.sqrt(noise_power / 2) * (
                np.random.randn(N, total_sampling_points) +
                1j * np.random.randn(N, total_sampling_points))
    X = np.outer(A, s_t) + noise
    R_xx = np.dot(X, X.conj().T) / total_sampling_points
    eigenvalues, eigenvectors = np.linalg.eig(R_xx)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    noise_subspace = eigenvectors[:, N - 1:]
    NN = noise_subspace @ noise_subspace.conj().T
    rankk = np.linalg.matrix_rank(NN)
    theta_scan = np.radians(np.arange(0, 90, 0.1))
    phi_scan = np.radians(np.arange(-90, 90, 0.1))
    music_spectrum = np.zeros((len(theta_scan), len(phi_scan)))

    for i, theta in enumerate(theta_scan):
        for j, phi in enumerate(phi_scan):
            a = np.exp(1j * delta_phi_n(n, N, D_lambda, theta, phi))
            vHa = np.dot(noise_subspace.conj().T, a)
            vHa = np.abs(vHa) ** 2
            music_spectrum[i, j] = 1 / vHa
    music_spectrum_dB = 10 * np.log10(music_spectrum / np.max(music_spectrum))
    Music_spectrum(phi_scan, theta_scan, music_spectrum_dB)

    max_idx = np.unravel_index(np.argmax(music_spectrum_dB, axis=None), music_spectrum_dB.shape)
    estimated_theta = theta_scan[max_idx[0]]
    estimated_phi = phi_scan[max_idx[1]]
    return np.rad2deg(estimated_theta), np.rad2deg(estimated_phi)

#Example
theta = 50
phi = -40
thetam, phim = DoaMusic(theta, phi, Fc, total_sampling_points, total_sampling_time, N, D_lambda, SNR)
print("Actual Theta: {0}, Actual Phi: {1}".format(theta, phi))
print("Estimated Theta: {0}, Estimated Phi: {1}".format(thetam, phim))

#Figures 3-17 and 3-18
def MUSICvsDoAfixedazimuth():
    theta = 30
    phis = np.arange(-60, 60, 2)
    mtheta = []
    dtheta = []
    mphi = []
    dphi = []

    for phi in phis:
        theta_m = 0
        phi_m = 0
        theta_d = 0
        phi_d = 0
        trials = 100
        for i in range(trials):
            thetaa1, phii1, thetaa2, phii2 = UCTMA.DoAEstimator(theta, phi, Fc, 1e10, total_sampling_time,
                     200000, N, D_lambda, 1e-6, 10, 1)
            thetam, phim = DoaMusic(theta, phi, Fc, 200000, total_sampling_time, N, D_lambda, 10)
            theta_m += ((thetam - theta) ** 2)
            theta_d += ((thetaa1 - theta) ** 2)
            phi_m += ((phim - phi) ** 2)
            phi_d += ((phii1 - phi)**2)
        dphi.append(phi_d / trials)
        mphi.append(phi_m / trials)
        mtheta.append(theta_m / trials)
        dtheta.append(theta_d / trials)

    plt.plot(phis, mtheta, label = 'MUSIC')
    plt.plot(phis, dtheta, label = 'UCTMA')
    plt.xlabel('Elevation angle')
    plt.ylabel('MSE of the azimuth angle')
    plt.title('MUSIC vs UCTMA under a fixed azimuth angle')
    plt.legend()
    plt.show()

    plt.plot(phis, mphi, label = 'MUSIC')
    plt.plot(phis, dphi, label = 'UCTMA')
    plt.xlabel('Elevation angle')
    plt.ylabel('MSE of the elevation angle')
    plt.title('MUSIC vs UCTMA under a fixed azimuth angle')
    plt.legend()
    plt.show()


#Figures 3-19 and 3-20
def MUSICvsDoAfixedelevation():
    phi = 30
    thetas = np.arange(25, 60, 3)
    mtheta = []
    dtheta = []
    mphi = []
    dphi = []

    for theta in thetas:
        theta_m = 0
        phi_m = 0
        theta_d = 0
        phi_d = 0
        trials = 200
        for i in range(trials):
            thetaa1, phii1, thetaa2, phii2 = UCTMA.DoAEstimator(theta, phi, Fc, 1e10, total_sampling_time,
                     200000, N, D_lambda, 1e-6, 10, 1)
            thetam, phim = DoaMusic(theta, phi, Fc, 200000, total_sampling_time, N, D_lambda, 10)
            theta_m += ((thetam - theta) ** 2)
            theta_d += ((thetaa1 - theta) ** 2)
            phi_m += ((phim - phi) ** 2)
            phi_d += ((phii1 - phi)**2)
        dphi.append(phi_d / trials)
        mphi.append(phi_m / trials)
        mtheta.append(theta_m / trials)
        dtheta.append(theta_d / trials)

    plt.plot(thetas, mtheta, label = 'MUSIC')
    plt.plot(thetas, dtheta, label = 'UCTMA')
    plt.xlabel('Azimuth angle')
    plt.ylabel('MSE of the azimuth angle')
    plt.title('MUSIC vs UCTMA under a fixed elevation angle')
    plt.legend()
    plt.show()

    plt.plot(thetas, mphi, label = 'MUSIC')
    plt.plot(thetas, dphi, label = 'UCTMA')
    plt.xlabel('Azimuth angle')
    plt.ylabel('MSE of the elevation angle')
    plt.title('MUSIC vs UCTMA under a fixed elevation angle')
    plt.legend()
    plt.show()


#Figures 3-21 and 3-22
def MUSICvsDoASNRs():
    theta = 15
    phi = 30
    SNRs = np.arange(-10, 21, 5)
    mtheta = []
    dtheta = []
    mphi = []
    dphi = []
    for SNR in SNRs:
        theta_m = 0
        phi_m = 0
        theta_d = 0
        phi_d = 0
        trials = 200
        realt = 0
        for i in range(trials):
            thetaa1, phii1, thetaa2, phii2 = UCTMA.DoAEstimator(theta, phi, Fc, 1e10, 2e-5,
                     1000000, N, D_lambda, 1e-6, SNR, 1)
            thetam, phim = DoaMusic(theta, phi, Fc, 1000000, 1e-5, N, D_lambda, SNR)

            # if ((thetam - theta) ** 2) > 5 or ((phim - phi) ** 2) > 5:
            #     continue
            theta_m += ((thetam - theta) ** 2)
            theta_d += ((thetaa1 - theta) ** 2)
            phi_m += ((phim - phi) ** 2)
            phi_d += ((phii1 - phi)**2)
            print(i)
        dphi.append(phi_d / trials)
        mphi.append(phi_m / trials)
        mtheta.append(theta_m / trials)
        dtheta.append(theta_d / trials)
        print(SNR)

    plt.plot(SNRs, mtheta, label = 'MUSIC')
    plt.plot(SNRs, dtheta, label = 'UCTMA')
    plt.xlabel('SNR(dB)')
    plt.ylabel('MSE of the azimuth angle')
    plt.title('MUSIC vs UCTMA under different SNRs(azimuth)')
    plt.legend()
    plt.show()

    plt.plot(SNRs, mphi, label = 'MUSIC')
    plt.plot(SNRs, dphi, label = 'UCTMA')
    plt.xlabel('SNR(dB)')
    plt.ylabel('MSE of the elevation angle')
    plt.title('MUSIC vs UCTMA under different SNRs(elevation)')
    plt.legend()
    plt.show()

