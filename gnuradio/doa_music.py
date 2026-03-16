#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lin
from gnuradio import gr


class doa_music(gr.sync_block):
    """
    2D MUSIC DOA estimation block for a uniform circular or linear array.
    Inputs: one already-decimated vector per antenna channel
    Output: flattened 2D spectrum (elevation_bins * azimuth_bins) in float32
    """

    def __init__(
        self,
        cpi_size=8192,               # now this is the vector length AT THE INPUT
        freq=433.0,
        array_dist=0.33,
        num_elements=5,
        array_type='UCA',
        azimuth_min=0.0,
        azimuth_max=359.0,
        azimuth_step_deg=1.0,
        elevation_min=0.0,
        elevation_max=90.0,
        elevation_step_deg=1.0,
        signal_dimension=2,
        diag_loading=1e-3,
    ):
        self.cpi_size = int(cpi_size)
        self.freq = float(freq)
        self.array_dist = float(array_dist)
        self.num_elements = int(num_elements)
        self.array_type = array_type
        self.signal_dimension = int(np.clip(signal_dimension, 1, num_elements - 1))
        self.diag_loading = float(max(diag_loading, 0.0))

        self.azimuth_min = float(azimuth_min)
        self.azimuth_max = float(azimuth_max)
        self.azimuth_step_deg = float(azimuth_step_deg)

        self.elevation_min = float(elevation_min)
        self.elevation_max = float(elevation_max)
        self.elevation_step_deg = float(elevation_step_deg)

        if self.azimuth_step_deg <= 0:
            raise ValueError("azimuth_step_deg must be > 0")
        if self.elevation_step_deg <= 0:
            raise ValueError("elevation_step_deg must be > 0")
        if self.azimuth_max < self.azimuth_min:
            raise ValueError("azimuth_max must be >= azimuth_min")
        if self.elevation_max < self.elevation_min:
            raise ValueError("elevation_max must be >= elevation_min")

        self.azimuths = np.arange(
            self.azimuth_min,
            self.azimuth_max + 0.5 * self.azimuth_step_deg,
            self.azimuth_step_deg,
            dtype=np.float32,
        )
        self.elevations = np.arange(
            self.elevation_min,
            self.elevation_max + 0.5 * self.elevation_step_deg,
            self.elevation_step_deg,
            dtype=np.float32,
        )

        self.azimuth_bins = len(self.azimuths)
        self.elevation_bins = len(self.elevations)
        self.output_len = self.azimuth_bins * self.elevation_bins

        gr.sync_block.__init__(
            self,
            name="DOA MUSIC",
            in_sig=[(np.complex64, self.cpi_size)] * self.num_elements,
            out_sig=[(np.float32, self.output_len)],
        )

        wavelength = 300.0 / self.freq  # meters if freq in MHz

        if self.array_type == 'UCA':
            self.array_radius_m = self.array_dist
        else:
            self.element_spacing_m = self.array_dist

        self.scanning_vectors = self.gen_scanning_vectors_2d(
            self.num_elements,
            wavelength,
            self.array_type,
            0.0,
        )

    def work(self, input_items, output_items):
        # Input is already decimated upstream.
        X = np.array(
            [input_items[i][0] for i in range(self.num_elements)],
            dtype=np.complex64
        )  # shape: (num_elements, cpi_size)

        R = self.corr_matrix(X)

        if self.diag_loading > 0.0:
            tr = np.trace(R).real / R.shape[0]
            R = R + (self.diag_loading * tr) * np.eye(R.shape[0], dtype=R.dtype)

        spectrum = self.DOA_MUSIC_2D(
            R,
            self.scanning_vectors,
            signal_dimension=self.signal_dimension
        )

        output_items[0][0][:] = self.DOA_plot_util_2d(spectrum)
        return 1

    def corr_matrix(self, X):
        N = X.shape[1]
        return (X @ X.conj().T) / max(N, 1)

    def gen_scanning_vectors_2d(self, M, wavelength, array_type, offset_deg=0.0):
        if array_type == "UCA":
            x = self.array_radius_m * np.cos(2 * np.pi / M * np.arange(M))
            y = -self.array_radius_m * np.sin(2 * np.pi / M * np.arange(M))
            z = np.zeros(M, dtype=np.float32)
        else:
            x = np.zeros(M, dtype=np.float32)
            y = -np.arange(M, dtype=np.float32) * self.element_spacing_m
            z = np.zeros(M, dtype=np.float32)

        scanning_vectors = np.zeros(
            (M, self.azimuth_bins, self.elevation_bins),
            dtype=np.complex64
        )

        for az_idx, az in enumerate(self.azimuths):
            az_rad = np.deg2rad(az + offset_deg)
            for el_idx, el in enumerate(self.elevations):
                el_rad = np.deg2rad(el)

                ux = np.cos(el_rad) * np.cos(az_rad)
                uy = np.cos(el_rad) * np.sin(az_rad)
                uz = np.sin(el_rad)

                phase = (2 * np.pi / wavelength) * (x * ux + y * uy + z * uz)
                scanning_vectors[:, az_idx, el_idx] = np.exp(1j * phase)

        return np.ascontiguousarray(scanning_vectors)

    def DOA_MUSIC_2D(self, R, scanning_vectors, signal_dimension=1):
        _, vi = lin.eigh(R)

        noise_dimension = R.shape[0] - signal_dimension
        E = vi[:, :noise_dimension]

        M = scanning_vectors.shape[0]
        A = scanning_vectors.reshape(M, -1)

        proj = E.conj().T @ A
        denom = np.sum(np.abs(proj) ** 2, axis=0)

        spectrum = 1.0 / np.maximum(denom, 1e-12)
        return spectrum.reshape(self.azimuth_bins, self.elevation_bins)

    def DOA_plot_util_2d(self, doa_data, log_scale_min=-100):
        doa_data = np.abs(doa_data)
        doa_data /= np.maximum(np.max(doa_data), 1e-12)
        doa_data = 10 * np.log10(np.maximum(doa_data, 1e-12))
        doa_data = np.maximum(doa_data, log_scale_min)

        # GUI contract: flattened as (elevation_bins, azimuth_bins)
        return doa_data.T.astype(np.float32).ravel()