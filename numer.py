import numpy as np
import matplotlib.pyplot as plt
import UCTMA

N = 4
Fc = 1e9  # Carrier frequency in Hz
Fs = 1e10  # Sampling frequency in Hz
total_sampling_time = 2e-5  # Total sampling time in seconds
Fm = 1e6  # Modulation frequency in Hz
D_lambda = 0.5  # D/lambda
total_sampling_points = 200000  # Total number of sampling points
Tp = 1e-6  # Period of the modulation function
SNR = 10  #Signal-to-Noise ratio
harm = -1
theta_true = 30
phi_true = 40
num_trials = 100
true_theta = 50
true_phi = -40

#Figures 3-1
def fixedtheta(theta):
    phis = np.arange(-80, 80, 3)
    mse_theta = []
    mse_phi = []
    trials = 1000
    for phi in phis:
        phi_mse = 0
        theta_mse = 0
        for i in range(trials):

            thetaa, phii, thetaa1, phii1 = UCTMA.DoAEstimator(theta, phi, Fc, Fs, total_sampling_time,
                 total_sampling_points, N, D_lambda, Tp, SNR, 1)

            phi_mse += (phi - phii)**2
            theta_mse+= (theta - thetaa) ** 2
        mse_phi.append(phi_mse/trials)
        mse_theta.append(theta_mse/trials)

    plt.plot(phis, mse_theta, label = 'Azimuth angle')
    plt.plot(phis, mse_phi, label = 'Elevation Angle')
    plt.xlabel('Elevation angle')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    
#Figure 3-2
def fixedphi(phi):
    thetas = np.arange(10, 60, 3)
    mse_theta = []
    mse_phi = []
    trials = 1000
    for theta in thetas:
        phi_mse = 0
        theta_mse = 0
        for i in range(trials):

            thetaa, phii, thetaa1, phii1 = UCTMA.DoAEstimator(theta, phi, Fc, Fs, total_sampling_time,
                 total_sampling_points, N, D_lambda, Tp, SNR, 1)

            phi_mse += (phi - phii)**2
            theta_mse+= (theta - thetaa) ** 2
        mse_phi.append(phi_mse/trials)
        mse_theta.append(theta_mse/trials)

    plt.plot(thetas, mse_theta, label = 'Azimuth angle')
    plt.plot(thetas, mse_phi, label = 'Elevation Angle')
    plt.xlabel('Azimuth angle')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    
#Figure 3-13
def MSEforDifferentSNRs():
    SNRs = np.arange(-10, 21, 5)  # SNR range from -10 dB to 20 dB
    monte_carlo_trials = 1000  # Number of trials per SNR
    mse_theta = []
    mse_phi = []
    for SNR in SNRs:
        theta_errors = []
        phi_errors = []
        for trial in range(monte_carlo_trials):
            theta_est1, phi_est1, theta_est2, phi_est2 = (UCTMA.DoAEstimator(theta_true, phi_true,
                                   Fc, Fs, 1e-5,
                                   1000000,4,
                                   0.5, 1e-7, SNR, 1))

            theta_error = (theta_est1 - theta_true) ** 2
            phi_error = (phi_est1 - phi_true) ** 2
            theta_errors.append(theta_error)
            phi_errors.append(phi_error)
        mse_theta.append(np.sqrt(np.mean(theta_errors)))
        mse_phi.append(np.sqrt(np.mean(phi_errors)))

    plt.plot(SNRs, mse_theta, label='MSE of Theta')
    plt.plot(SNRs, mse_phi, label='MSE of Phi')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE (degrees^2)')
    plt.title('MSE of DoA Estimation vs. SNR')
    plt.legend()
    plt.grid(True)
    plt.show()

#Figure 3-14
def MSEforDifeerentApertures():
    Ds = np.arange(0.2, 0.6, 0.05)
    monte_carlo_trials = 200
    mse_theta = []
    mse_phi = []
    for D in Ds:
        theta_errors = []
        phi_errors = []
        for trial in range(monte_carlo_trials):
            theta_est, phi_est, theta_est1, phi_est1 = (UCTMA.
                                  DoAEstimator(theta_true, phi_true,
                                               Fc, Fs, total_sampling_time,
                                               total_sampling_points, 4,
                                               D, Tp, SNR, harm))
            # Calculate squared errors
            theta_error = (theta_est - theta_true) ** 2
            phi_error = (phi_est - phi_true) ** 2
            theta_errors.append(theta_error)
            phi_errors.append(phi_error)
        # Calculate MSE for the current D
        mse_theta.append(np.mean(theta_errors))
        mse_phi.append(np.mean(phi_errors))
    # Plotting the MSEs
    plt.plot(Ds, mse_theta, label='MSE of Theta')
    plt.plot(Ds, mse_phi, label='MSE of Phi')
    plt.xlabel('D/λ ')
    plt.ylabel('MSE (degrees^2)')
    plt.title('MSE of DoA Estimation vs. D/lambda')
    plt.legend()
    plt.grid(True)
    plt.show()


#Figure 3-15
def MSEforDifferentNumberOfSnapshots():
    snaps = np.arange(300000, 3500001, 300000)
    monte_carlo_trials = 1000  # Number of trials per SNR
    mse_theta = []
    mse_phi = []
    for snap in snaps:
        theta_errors = []
        phi_errors = []
        for trial in range(monte_carlo_trials):
            theta_est1, phi_est1, theta_est2, phi_est2 = UCTMA.DoAEstimator (theta_true, phi_true,
                                     Fc, Fs, total_sampling_time,
                                     snap, 4,
                                     0.5, Tp, 15, 1)
            # Calculate squared errors
            theta_error = (theta_est1 - theta_true) ** 2
            phi_error = (phi_est1 - phi_true) ** 2
            theta_errors.append(theta_error)
            phi_errors.append(phi_error)

        # Calculate MSE for the current SNR
        mse_theta.append(np.sqrt(np.mean(theta_errors)))
        mse_phi.append(np.sqrt(np.mean(phi_errors)))

    plt.plot(snaps, mse_theta, label='MSE of Theta')
    plt.plot(snaps, mse_phi, label='MSE of Phi')
    plt.xlabel('Number of Snapshots')
    plt.ylabel('MSE (degrees^2)')
    plt.title('MSE of DoA Estimation vs. Number of Snapshots')
    plt.legend()
    plt.grid(True)
    plt.show()
    
#Figures 4-1 and 4-2
def fixedthetan1p1(theta):
    phis = np.arange(-80, 80, 3)
    trials = 100
    mse_theta_p1 = []
    mse_phi_p1 = []
    mse_theta_n1 = []
    mse_phi_n1 = []
    for phi in phis:
        phi_mse_p1 = 0
        theta_mse_p1 = 0
        phi_mse_n1 = 0
        theta_mse_n1 = 0
        for i in range(trials):
            thetaa, phii, thetaa1, phii1 = UCTMA.DoAEstimator(theta, phi, Fc, Fs, total_sampling_time,
                 total_sampling_points, N, D_lambda, Tp, SNR, 1)

            phi_mse_p1 += (phi - phii)**2
            theta_mse_p1+= (theta - thetaa) ** 2
            phi_mse_n1 += (phi - phii1) ** 2
            theta_mse_n1 += (theta - thetaa1) ** 2
        mse_phi_p1.append(phi_mse_p1/trials)
        mse_theta_p1.append(theta_mse_p1/trials)
        mse_phi_n1.append(phi_mse_n1 / trials)
        mse_theta_n1.append(theta_mse_n1 / trials)
    plt.plot(phis, mse_theta_p1, label = '+1st harmonic')
    plt.plot(phis, mse_theta_n1, label = '-1st harmonic')
    plt.xlabel('Elevation angle')
    plt.ylabel('Mean Squared Error(Azimuth)')
    plt.title('MSE of the azimuth angle(comparing the two algorithms)')
    plt.legend()
    plt.show()

    plt.plot(phis, mse_phi_p1, label='+1st harmonic')
    plt.plot(phis, mse_phi_n1, label='-1st harmonic')
    plt.xlabel('Elevation angle')
    plt.ylabel('Mean Squared Error(Elevation)')
    plt.title('MSE of the elevation angle(comparing the two algorithms)')
    plt.legend()
    plt.show()

#Figures 4-3 and 4-4
def fixedphin1p1(phi):
    thetas = np.arange(15, 60, 3)
    trials = 100
    mse_theta_p1 = []
    mse_phi_p1 = []
    mse_theta_n1 = []
    mse_phi_n1 = []
    for theta in thetas:
        phi_mse_p1 = 0
        theta_mse_p1 = 0
        phi_mse_n1 = 0
        theta_mse_n1 = 0
        for i in range(trials):

            thetaa, phii, thetaa1, phii1 = UCTMA.DoAEstimator(theta, phi, Fc, Fs, total_sampling_time,
                 total_sampling_points, N, D_lambda, Tp, SNR, 1)

            phi_mse_p1 += (phi - phii)**2
            theta_mse_p1+= (theta - thetaa) ** 2
            phi_mse_n1 += (phi - phii1) ** 2
            theta_mse_n1 += (theta - thetaa1) ** 2
        mse_phi_p1.append(phi_mse_p1/trials)
        mse_theta_p1.append(theta_mse_p1/trials)
        mse_phi_n1.append(phi_mse_n1 / trials)
        mse_theta_n1.append(theta_mse_n1 / trials)

    plt.plot(thetas, mse_theta_p1, label = '+1st harmonic')
    plt.plot(thetas, mse_theta_n1, label = '-1st harmonic')
    plt.xlabel('Azimuth angle')
    plt.ylabel('Mean Squared Error(Azimuth)')
    plt.title('MSE of the azimuth angle(comparing the two algorithms)')
    plt.legend()
    plt.show()

    plt.plot(thetas, mse_phi_p1, label='+1st harmonic')
    plt.plot(thetas, mse_phi_n1, label='-1st harmonic')
    plt.xlabel('Azimuth angle')
    plt.ylabel('Mean Squared Error(Elevation)')
    plt.title('MSE of the elevation angle(comparing the two algorithms)')
    plt.legend()
    plt.show()

#Figures 4-5, 4-6 and 4-7
def n1p1MSEdiffSNR():
    theta = 30
    phi = 40
    SNRs = np.arange(-10, 21, 5)
    theta11 = []
    theta22 = []
    thetadd = []
    phi11 = []
    phi22 = []
    phidd = []
    for SNR in SNRs:
        trials = 100
        theta1s = 0
        theta2s = 0
        phi1s = 0
        phi2s = 0
        thetad = 0
        phid = 0
        for _ in range(trials):
            theta1, phi1, theta2, phi2 = UCTMA.DoAEstimator(theta, phi, Fc, Fs, total_sampling_time,
                         total_sampling_points, N, D_lambda, Tp, SNR, 1)
            theta1s += (theta1 - theta)**2/trials
            theta2s += (theta2 - theta)**2/trials
            thetad += (theta1 - theta2)**2/trials
            phi1s += (phi1 - phi)**2/trials
            phi2s += (phi2 - phi)**2/trials
            phid += (phi1 - phi2)**2/trials

        theta11.append(theta1s)
        theta22.append(theta2s)
        phi11.append(phi1s)
        phi22.append(phi2s)
        thetadd.append(thetad)
        phidd.append(phid)

    plt.plot(SNRs, theta11, label = '+1st harmonic')
    plt.plot(SNRs, theta22, label = '-1st harmonic')
    plt.xlabel('SNRs')
    plt.ylabel('MSE of azimuth angles')
    plt.title('MSE of the azimuth angle(different estimation algorithms) vs SNR')
    plt.legend()
    plt.show()

    plt.plot(SNRs, phi11, label = '+1st harmonic')
    plt.plot(SNRs, phi22, label = '-1st harmonic')
    plt.xlabel('SNRs')
    plt.ylabel('MSE of elevation angle')
    plt.title('MSE of the elevation angle(different estimation algorithms) vs SNR')
    plt.legend()
    plt.show()

    plt.plot(SNRs, thetadd, label = 'Azimuth')
    plt.plot(SNRs, phidd, label = 'Elevation')
    plt.xlabel('SNRs')
    plt.ylabel('MSE of two estimated angles')
    plt.title('MSE of the azimuth and elevation angles(+1st harmonic and -1st harmonic) vs SNR')
    plt.legend()
    plt.show()

def realtiveratiop1n1():
    thetas = np.arange(1, 15, 1)
    phi = 40
    ps = []
    qs = []
    pns = []
    qns = []
    for theta in thetas:
        mmm1, mmm2 = UCTMA.DoAEstimator(theta, phi, Fc, Fs, total_sampling_time,
                                  total_sampling_points, N, 0.5, 1e-6, 10)
        mmm = UCTMA.pandq(theta, phi, D_lambda)
        p1s = np.abs(np.abs(mmm1.real) - np.abs(mmm.real)) / np.abs(mmm.real)
        p2s = np.abs(np.abs(mmm2.real) - np.abs(mmm.real)) / np.abs(mmm.real)
        q1s = np.abs(np.abs(mmm1.imag) - np.abs(mmm.imag)) / np.abs(mmm.imag)
        q2s = np.abs(np.abs(mmm2.imag) - np.abs(mmm.imag)) / np.abs(mmm.imag)
        ps.append(p1s)
        pns.append(p2s)
        qs.append(q1s)
        qns.append(q2s)

    plt.plot(thetas, ps, label='Real part')
    plt.plot(thetas, qs, label='Imaginary part')
    plt.xlabel('Azimuth angle')
    plt.ylabel('Relative Error')
    plt.title('Relative errors of real and imag. parts(+1st harmonic)')
    plt.legend()
    plt.show()

    plt.plot(thetas, pns, label='Real part')
    plt.plot(thetas, qns, label='Imaginary part')
    plt.xlabel('Azimuth angle')
    plt.ylabel('Relative Error')
    plt.title('Relative errors of real and imag. parts(-1st harmonic)')
    plt.legend()
    plt.show()

def fixedtheta1(theta, ax):
    phi_scan = np.arange(-80, 80, 3)
    phi1 = []
    trials = 100
    for phi in phi_scan:
        phi_mean1 = 0
        phi_mean2 = 0
        for numm in range(100):
            thetaa1, phii1 = (UCTMA
                            .DoAEstimator(theta, phi, Fc,
                                          100e9, 1e-5,
                                          1000000, 4,
                                          0.5, 2e-7, 10, 1))
            phi_mean1 +=(phii1 - phi)**2
        phi1.append(phi_mean1/trials)
    ax.plot(phi_scan, phi1, label='Using +1st harmonic')
    #ax.plot(phi_scan, phi2, label='Using -1st harmonic')
    ax.set_title('θ = {}°'.format(theta))
    ax.set_xlabel('Elevation angle (°)')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True)

#Figure 4-13
def fixeddiffthetas():
    fig, axs = plt.subplots(3, 2, figsize=(10, 6)) # Adjusted figure size for clarity
    thetas = [1, 2, 3, 8, 9, 10]
    axs_flat = axs.flatten()
    for i, theta in enumerate(thetas):
        fixedtheta1(theta, axs_flat[i])
    for i in range(len(thetas), len(axs_flat)):
        fig.delaxes(axs_flat[i])
    plt.tight_layout()
    plt.show()

