#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lin
from scipy import signal
from gnuradio import gr


class doa_music(gr.sync_block):
    """
    2D MUSIC DOA estimation block for a uniform circular or linear array.
    Outputs a flattened 2D spectrum (azimuth_bins * elevation_bins) in float32.
    """

    def __init__(
        self,
        cpi_size=2**20,
        freq=433.0,
        array_dist=0.33,
        num_elements=5,
        array_type='UCA',
        azimuth_bins=360,
        elevation_bins=91,
        elevation_min=0.0,
        elevation_max=90.0,
        decimation=128,
        signal_dimension=2,          # increased from implicit 1
        diag_loading=1e-3,           # small covariance loading
    ):
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max

        self.output_len = azimuth_bins * elevation_bins

        gr.sync_block.__init__(
            self,
            name="DOA MUSIC",
            in_sig=[(np.complex64, cpi_size)] * num_elements,
            out_sig=[(np.float32, self.output_len)]
        )

        self.cpi_size = cpi_size
        self.freq = freq
        self.array_dist = array_dist
        self.num_elements = num_elements
        self.array_type = array_type
        self.decimation = decimation
        self.signal_dimension = int(np.clip(signal_dimension, 1, num_elements - 1))
        self.diag_loading = float(max(diag_loading, 0.0))

        wavelength = 300 / freq  # meters if freq in MHz

        # Keep physical geometry in meters
        if array_type == 'UCA':
            self.array_radius_m = array_dist   # array_dist is radius in meters
        else:
            self.element_spacing_m = array_dist

        self.scanning_vectors = self.gen_scanning_vectors_2d(
            self.num_elements,
            wavelength,
            self.array_type,
            0,
            self.azimuth_bins,
            self.elevation_bins,
            self.elevation_min,
            self.elevation_max,
        )

        #print("DOA MUSIC initialized: spacing / wavelength =", spacing_mult)

    def work(self, input_items, output_items):
        # Stack inputs into (num_elements, cpi_size)
        processed_signal = np.array([input_items[i][0] for i in range(self.num_elements)], dtype=np.complex64)

        # Decimate along time axis
        decimated = signal.decimate(processed_signal, self.decimation, axis=1, ftype='fir')

        # Compute spatial covariance matrix
        R = self.corr_matrix(decimated)

        # Diagonal loading for stability in multipath / mismatch conditions
        if self.diag_loading > 0.0:
            tr = np.trace(R).real / R.shape[0]
            R = R + (self.diag_loading * tr) * np.eye(R.shape[0], dtype=R.dtype)

        # Compute 2D MUSIC spectrum (use configured signal dimension)
        spectrum = self.DOA_MUSIC_2D(R, self.scanning_vectors, signal_dimension=self.signal_dimension)

        # Flatten and convert to float32 for GUI
        spectrum_plot = self.DOA_plot_util_2d(spectrum)

        output_items[0][0][:] = spectrum_plot
        return 1

    def corr_matrix(self, X):
        """Compute the sample covariance matrix."""
        N = X.shape[1]
        R = np.dot(X, X.conj().T) / N
        return R

    def gen_scanning_vectors_2d(
        self,
        M,
        wavelength,
        array_type,
        offset,
        azimuth_bins,
        elevation_bins,
        elevation_min,
        elevation_max,
    ):
        azimuths = np.linspace(0, 359, azimuth_bins)
        elevations = np.linspace(elevation_min, elevation_max, elevation_bins)

        if array_type == "UCA":
            # Coordinates in meters
            x = self.array_radius_m * np.cos(2 * np.pi / M * np.arange(M))
            y = -self.array_radius_m * np.sin(2 * np.pi / M * np.arange(M))
            z = np.zeros(M)
        else:
            x = np.zeros(M)
            y = -np.arange(M) * self.element_spacing_m
            z = np.zeros(M)

        scanning_vectors = np.zeros((M, azimuth_bins, elevation_bins), dtype=np.complex64)

        for az_idx, az in enumerate(azimuths):
            az_rad = np.deg2rad(az + offset)
            for el_idx, el in enumerate(elevations):
                el_rad = np.deg2rad(el)
                ux = np.cos(el_rad) * np.cos(az_rad)
                uy = np.cos(el_rad) * np.sin(az_rad)
                uz = np.sin(el_rad)
                phase = (2 * np.pi / wavelength) * (x * ux + y * uy + z * uz)
                scanning_vectors[:, az_idx, el_idx] = np.exp(1j * phase)

        return np.ascontiguousarray(scanning_vectors)

    def DOA_MUSIC_2D(self, R, scanning_vectors, signal_dimension=1):
        """Compute the 2D MUSIC spectrum."""
        sigmai, vi = lin.eigh(R)
        idx = sigmai.argsort()
        vi = vi[:, idx]

        noise_dimension = R.shape[0] - signal_dimension
        E = vi[:, :noise_dimension]
        E_ct = E @ E.conj().T

        M = scanning_vectors.shape[0]
        A = scanning_vectors.reshape(M, -1)
        denom = np.sum(A.conj() * (E_ct @ A), axis=0)
        spectrum = 1.0 / np.maximum(np.abs(denom), 1e-12)
        return spectrum.reshape(self.azimuth_bins, self.elevation_bins)

    def DOA_plot_util_2d(self, doa_data, log_scale_min=-100):
        """Prepare data for display (dB scale).

        Output format contract for GUI:
        flattened as (elevation_bins, azimuth_bins) in C-order.
        """
        doa_data = np.abs(doa_data)
        doa_data /= np.maximum(np.max(doa_data), 1e-12)
        doa_data = 10 * np.log10(np.maximum(doa_data, 1e-12))
        doa_data = np.maximum(doa_data, log_scale_min)

        # doa_data is (azimuth_bins, elevation_bins) -> transpose to (elevation_bins, azimuth_bins)
        return doa_data.T.astype(np.float32).ravel()
