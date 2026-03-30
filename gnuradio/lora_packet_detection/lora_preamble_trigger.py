#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr


class lora_preamble_trigger(gr.sync_block):
    """Coarse LoRa preamble trigger.

    This block monitors a single complex stream and emits a float trigger pulse
    when it sees `min_repeats` consecutive LoRa-like upchirps. The detector is
    intentionally lightweight and is meant to provide a packet-start trigger for
    a downstream gated DOA capture, not a full LoRa synchronizer/decoder.

    Detection method:
      - Break the stream into symbol-sized windows.
      - Dechirp each window with a locally generated downchirp.
      - FFT the dechirped symbol.
      - During the LoRa preamble, repeated upchirps collapse to the same FFT bin.
      - If the dominant bin repeats with sufficient confidence for
        `min_repeats` windows, emit one trigger pulse.

    Notes:
      - Best results require fixed BW/SF and a post-filter sample rate chosen so
        samples_per_symbol is close to an integer.
      - This is a coarse detector. You may need to tune `bin_tolerance`,
        `peak_ratio_threshold`, and `trigger_advance_symbols` on real captures.
    """

    def __init__(
        self,
        samp_rate=600000.0,
        bandwidth=250000.0,
        spreading_factor=7,
        min_repeats=6,
        bin_tolerance=1,
        peak_ratio_threshold=4.0,
        trigger_advance_symbols=5,
        refractory_symbols=12,
    ):
        gr.sync_block.__init__(
            self,
            name="lora_preamble_trigger",
            in_sig=[np.complex64],
            out_sig=[np.float32],
        )

        self.samp_rate = float(samp_rate)
        self.bandwidth = float(bandwidth)
        self.spreading_factor = int(spreading_factor)
        self.min_repeats = int(min_repeats)
        self.bin_tolerance = int(bin_tolerance)
        self.peak_ratio_threshold = float(peak_ratio_threshold)
        self.trigger_advance_symbols = int(trigger_advance_symbols)
        self.refractory_symbols = int(refractory_symbols)

        self.symbol_time = (2 ** self.spreading_factor) / self.bandwidth
        self.samples_per_symbol = int(round(self.samp_rate * self.symbol_time))
        self.fft_len = self.samples_per_symbol

        n = np.arange(self.samples_per_symbol, dtype=np.float32)
        # Local downchirp for dechirping an upchirp-like preamble symbol.
        # This is a coarse discrete model that works reasonably well once the
        # target signal has been filtered close to DC.
        self.downchirp = np.exp(-1j * np.pi * n * n / self.samples_per_symbol).astype(np.complex64)

        self._buf = np.empty(0, dtype=np.complex64)
        self._sample_count = 0
        self._last_peak_bin = None
        self._repeat_count = 0
        self._cooldown_samples = 0
        self._scheduled_trigger_abs = None

    def _analyze_symbol(self, sym):
        dechirped = sym * self.downchirp
        spec = np.fft.fft(dechirped, n=self.fft_len)
        mag = np.abs(spec)
        peak_bin = int(np.argmax(mag))
        peak_val = float(mag[peak_bin])
        mean_val = float(np.mean(mag)) + 1e-12
        peak_ratio = peak_val / mean_val
        return peak_bin, peak_ratio

    def work(self, input_items, output_items):
        x = np.asarray(input_items[0], dtype=np.complex64)
        y = output_items[0]
        y[:] = 0.0

        start_abs = self._sample_count
        self._buf = np.concatenate((self._buf, x))

        while self._cooldown_samples > 0 and len(self._buf) > 0:
            step = min(self._cooldown_samples, len(self._buf))
            self._buf = self._buf[step:]
            self._sample_count += step
            self._cooldown_samples -= step

        while len(self._buf) >= self.samples_per_symbol:
            sym = self._buf[:self.samples_per_symbol]
            peak_bin, peak_ratio = self._analyze_symbol(sym)

            matched = False
            if self._last_peak_bin is not None:
                delta = abs(peak_bin - self._last_peak_bin)
                delta = min(delta, self.fft_len - delta)
                matched = delta <= self.bin_tolerance

            if peak_ratio >= self.peak_ratio_threshold:
                if matched:
                    self._repeat_count += 1
                else:
                    self._repeat_count = 1
                self._last_peak_bin = peak_bin
            else:
                self._repeat_count = 0
                self._last_peak_bin = None

            if self._repeat_count >= self.min_repeats:
                trigger_at = self._sample_count - self.trigger_advance_symbols * self.samples_per_symbol
                if trigger_at < 0:
                    trigger_at = 0
                self._scheduled_trigger_abs = trigger_at
                self._repeat_count = 0
                self._last_peak_bin = None
                self._cooldown_samples = self.refractory_symbols * self.samples_per_symbol

            self._buf = self._buf[self.samples_per_symbol:]
            self._sample_count += self.samples_per_symbol

        end_abs = start_abs + len(x)
        if self._scheduled_trigger_abs is not None and start_abs <= self._scheduled_trigger_abs < end_abs:
            idx = int(self._scheduled_trigger_abs - start_abs)
            if 0 <= idx < len(y):
                y[idx] = 1.0
            self._scheduled_trigger_abs = None

        return len(x)
