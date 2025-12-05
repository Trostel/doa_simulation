import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import os

# from numba import njit, prange
import time

"""
LoRa Symbol Generation
"""
C = 3e8 # speed of light
# Kraken sampling frequency
SAMPLING_FREQUENCY = 2.56e6 # 2.56 MHz
SAMPLING_PERIOD = 1 / SAMPLING_FREQUENCY

# LoRa
SCALING_FACTOR = 7
SPREAD_SPECTRUM_BANDWIDTH = 500e3 # 500 KHz
M = 2 ** SCALING_FACTOR # Num. of spread frequencies around center
SYMBOL_DURATION = (2 ** SCALING_FACTOR)/SPREAD_SPECTRUM_BANDWIDTH
SAMPLES_PER_SYMBOL = int(SAMPLING_FREQUENCY * SYMBOL_DURATION)

# Transmission
CARRIER_FREQUENCY = 903e6 # 903 MHz

PAYLOAD = "Hello, World!abcd1234567890qwertyuiopasdfghjk;lzxcvbnm," # Message to transmit via lora!

# Generate upchirp 
# To be used for symbol generation
def upchirp():
    t = np.arange(SAMPLES_PER_SYMBOL) / SAMPLING_FREQUENCY # Timestep of a sample
    f0 = -SPREAD_SPECTRUM_BANDWIDTH/2
    phase = 2 * np.pi * (f0 * t + 0.5 * (SPREAD_SPECTRUM_BANDWIDTH / (SYMBOL_DURATION)) * t**2)
    return np.exp(1j * phase)

def generate_lora_symbol(symbol):
    chirp = upchirp()
    phase_shift = int(symbol * len(chirp) / M)
    return np.roll(chirp, phase_shift)

def modulate(payload):
    preamble = np.concatenate([upchirp() for i in range(8)]) # 8 unshifted upchirps 
    symbols = [ord(c) % M for c in payload] # converting our payload into mod M for encoding
    payload_chirps = np.concatenate([
        generate_lora_symbol(s) for s in symbols]) # Shift chirps based on symbol to generate lora symbols
    return np.concatenate([preamble, payload_chirps])

"""
UCA Setup and DOA
"""

NUM_ANTENNAS = 5
UCA_RADIUS = 0.133 # meters
TRUE_SIGNAL_DOA = [30, 25] # [elevation, azimuth]
WAVELENGTH = C / CARRIER_FREQUENCY
K = 2 * np.pi / WAVELENGTH # Signal wavenumber
ANTENNA_ANGLES = np.linspace(0, 2 * np.pi, NUM_ANTENNAS, endpoint=False)

# For noise calculation
NOISE_TEMPERATURE = 293 # deg kelvin
SYSTEM_RESISTANCE = 50 # ohms

# For FSPL
DISTANCE = 2000 # meters; to become time variable
EXPONENT = 2

# DOA Search Area
ELEVATION_RANGE_DEG = 20 # +/- /2 of this value
AZIMUTH_RANGE_DEG = 20

# Processing parallelization
NUM_CORES = os.cpu_count()
PROCESS_POOL : concurrent.futures.Executor = None

# theta -> elevation, phi -> azimuth
# for UCA
#@njit
def generate_steering_vectors(theta_deg, phi_deg): 
    THETA = np.deg2rad(theta_deg)
    PHI = np.deg2rad(phi_deg)
    return np.exp(1j * K * UCA_RADIUS * np.sin(THETA) * np.cos(PHI - ANTENNA_ANGLES))
    
def precompute_steering_vectors(theta_scan, phi_scan):
    THETA = np.deg2rad(theta_scan)[:, None, None]  # (T,1,1)
    PHI = np.deg2rad(phi_scan)[None, :, None]      # (1,P,1)
    ANT = ANTENNA_ANGLES[None, None, :]            # (1,1,N)

    A = np.exp(1j * K * UCA_RADIUS * np.sin(THETA) * np.cos(PHI - ANT))
    # shape: (T, P, N)
    return A

# Returns X, the received signal
def receive_signal(tx_iq):
    STEERING_VECTORS = generate_steering_vectors(TRUE_SIGNAL_DOA[0], TRUE_SIGNAL_DOA[1])
    return STEERING_VECTORS[:, np.newaxis] @ tx_iq[np.newaxis, :] 

# Remember to account for FSPL BEFORE gaussian noise!
def add_FSPL(signal):
    FSPL_LINEAR = (4 * np.pi * DISTANCE / WAVELENGTH) ** EXPONENT
    return signal / np.sqrt(FSPL_LINEAR)
# We can add gaussian noise independently to the measurements from each antenna since 
# gaussian noise's autocorrelation = 0!
def add_gaussian_noise(signal):
    BOLTZMANN_CONSTANT = 1.38064852e-23
    # thermal noise power in watts
    N_0 = BOLTZMANN_CONSTANT * NOISE_TEMPERATURE * SPREAD_SPECTRUM_BANDWIDTH
    ### ignoring receiver noise figure

    # RMS voltage
    N_0_RMS = np.sqrt(N_0 * SYSTEM_RESISTANCE)
    
    noise = N_0_RMS / np.sqrt(2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

def DOA_2D_MUSIC(X):
    # Compute covariance
    R = (X @ X.conj().T) / X.shape[1]
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    En = eigvecs[:, 1:]  # noise subspace
    
    # Scan grid
    theta_scan = np.linspace(0, 90, 91)
    phi_scan = np.linspace(0, 360, 361)
    
    # Precompute steering vectors
    A = precompute_steering_vectors(theta_scan, phi_scan)

    Pmusic = compute_Pmusic(A, En)    
    # Find peak
    idx_peak = np.unravel_index(np.argmax(Pmusic), Pmusic.shape)
    theta_est = theta_scan[idx_peak[0]]
    phi_est = phi_scan[idx_peak[1]]
    
    return theta_est, phi_est

def compute_Pmusic(A, En):
    T, P, N = A.shape
    Pmusic = np.zeros((T, P))
    EnEnH = En @ En.conj().T
    TP, N = A.shape[0]*A.shape[1], A.shape[2]
    A_reshaped = A.reshape(TP, N)  # shape (T*P, N)
    denom = np.sum(np.conj(A_reshaped) @ EnEnH * A_reshaped, axis=1)
    Pmusic = (1 / np.abs(denom)).reshape(A.shape[0], A.shape[1])
    return Pmusic


def PLOT_MUSIC_SPECTRUM(Pmusic_dB, phi_scan, theta_scan):
    theta_true, phi_true = TRUE_SIGNAL_DOA

    peak_idx = np.unravel_index(np.argmax(Pmusic_dB), Pmusic_dB.shape)
    theta_est = theta_scan[peak_idx[0]]
    phi_est = phi_scan[peak_idx[1]]
    
    print(f"True DOA: Elevation={theta_true} deg, Azimuth={phi_true} deg")
    print(f"Estimated DOA: Elevation={theta_est:.2f} deg, Azimuth={phi_est:.2f} deg")

    plt.figure(figsize=(8,6))
    plt.imshow(Pmusic_dB, extent=[phi_scan[0], phi_scan[-1], theta_scan[0], theta_scan[-1]],
            origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='MUSIC Spectrum (dB)')
    plt.title(f"2D MUSIC Spectrum (True: theta={theta_true} deg, phi={phi_true} deg)")
    plt.xlabel("Azimuth phi (degrees)")
    plt.ylabel("Elevation theta (degrees)")
    plt.scatter([phi_est], [theta_est], c='white', marker='x', s=100, label='Estimated DOA')
    plt.scatter([phi_true], [theta_true], c='green', alpha=0.2, marker='o', s=20, label='True DOA')
    plt.legend()
    plt.show()

"""
Flight path
"""
# Vertical path
# z(t) = -2.8t(t-155){ 0 < t < 20 }
# z`(t) = -54t + 8640 { 20 <= t < 160 }

# Horizontal path
# x(t) = 160 - t
# y(t) = t
FLIGHT_TIME = 160 # seconds
NUM_SAMPLES = FLIGHT_TIME/SAMPLING_PERIOD

# Must adjust p(t) based on total flight time
def parabolic_trajectory():
    NUM_SAMPLES_INT = int(NUM_SAMPLES)
    x = np.linspace(160, 0, NUM_SAMPLES_INT)
    y = np.linspace(0, 160, NUM_SAMPLES_INT)
    t1 = np.linspace(0, 20, int(NUM_SAMPLES_INT * 20/160))
    t2 = np.linspace(20, 160, int(NUM_SAMPLES_INT * 140/160))
    z = np.concatenate([-2.8*t1*(t1-155), -54*t2 + 8640])
    return np.stack([x, y, z], axis=0)

def simulate_flight():
    for t in range(0, FLIGHT_TIME, SAMPLING_PERIOD):
        pass

if __name__ == "__main__":
    #print(parabolic_trajectory().shape)
    tx_iq = modulate(PAYLOAD)
    rx_signal = receive_signal(tx_iq)
    rx_signal_FSPL = add_FSPL(rx_signal)
    rx_signal_FSPL_WN = add_gaussian_noise(rx_signal_FSPL)

    #before = time.perf_counter()
    DOA_2D_MUSIC(rx_signal_FSPL_WN)
    #seq_time = time.perf_counter() - before

    #print(f"Seq_time: {seq_time}")
    
    Pmusic_dB, phi_scan = DOA_2D_MUSIC(rx_signal_FSPL_WN)

    PLOT_MUSIC_SPECTRUM(Pmusic_dB, phi_scan, theta_scan)