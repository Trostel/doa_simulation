#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr


class triggered_vectorizer(gr.basic_block):
    """Capture a fixed sample window from multiple channels after a trigger.

    Inputs:
      - num_channels complex streams
      - 1 float trigger stream; values > trigger_threshold start a capture

    Outputs:
      - num_channels complex vectors of length window_len

    The block preserves only one active capture at a time. New triggers are
    ignored while a capture is in progress. This is exactly what we want for
    burst-triggered DOA snapshots.
    """

    def __init__(
        self,
        num_channels=5,
        window_len=2048,
        delay_after_trigger=0,
        trigger_threshold=0.5,
        rearm_samples=0,
    ):
        self.num_channels = int(num_channels)
        self.window_len = int(window_len)
        self.delay_after_trigger = int(delay_after_trigger)
        self.trigger_threshold = float(trigger_threshold)
        self.rearm_samples = int(rearm_samples)

        gr.basic_block.__init__(
            self,
            name="triggered_vectorizer",
            in_sig=[np.complex64] * self.num_channels + [np.float32],
            out_sig=[(np.complex64, self.window_len)] * self.num_channels,
        )

        self._capture_remaining = 0
        self._delay_remaining = 0
        self._cooldown_remaining = 0
        self._capturing = False
        self._capture_buffers = [np.empty(0, dtype=np.complex64) for _ in range(self.num_channels)]

    def forecast(self, noutput_items, ninput_items_required):
        # We only ever emit at most one vector per channel per call.
        # Request enough scalar samples to make progress on a capture.
        need = max(self.window_len + self.delay_after_trigger, 256)
        for i in range(self.num_channels + 1):
            ninput_items_required[i] = need

    def general_work(self, input_items, output_items):
        chans = [np.asarray(input_items[i], dtype=np.complex64) for i in range(self.num_channels)]
        trig = np.asarray(input_items[self.num_channels], dtype=np.float32)

        n = min(len(trig), *(len(c) for c in chans))
        if n == 0:
            return 0

        produced = False
        consume_n = n
        i = 0

        while i < n and not produced:
            if self._cooldown_remaining > 0:
                step = min(self._cooldown_remaining, n - i)
                self._cooldown_remaining -= step
                i += step
                continue

            if not self._capturing:
                hits = np.flatnonzero(trig[i:n] > self.trigger_threshold)
                if hits.size == 0:
                    i = n
                    break
                hit_idx = i + int(hits[0])
                self._capturing = True
                self._delay_remaining = self.delay_after_trigger
                self._capture_remaining = self.window_len
                self._capture_buffers = [np.empty(0, dtype=np.complex64) for _ in range(self.num_channels)]
                i = hit_idx

            if self._capturing:
                if self._delay_remaining > 0:
                    step = min(self._delay_remaining, n - i)
                    self._delay_remaining -= step
                    i += step
                    if i >= n:
                        break

                take = min(self._capture_remaining, n - i)
                if take > 0:
                    for ch in range(self.num_channels):
                        self._capture_buffers[ch] = np.concatenate((self._capture_buffers[ch], chans[ch][i:i+take]))
                    self._capture_remaining -= take
                    i += take

                if self._capture_remaining == 0:
                    for ch in range(self.num_channels):
                        output_items[ch][0][:] = self._capture_buffers[ch][:self.window_len]
                    produced = True
                    self._capturing = False
                    self._cooldown_remaining = self.rearm_samples
                    consume_n = i
                    break

        if not produced:
            consume_n = i if i > 0 else n

        for inp in range(self.num_channels + 1):
            self.consume(inp, consume_n)

        return 1 if produced else 0
