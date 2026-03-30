#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gnuradio import gr
from gnuradio import filter
from gnuradio import fft
from gnuradio.filter import firdes
import sys
import signal
import threading

from gnuradio import krakensdr
import gnuradio.lora_packet_detection.doa_music_lora as doa_music
from lora_preamble_trigger import lora_preamble_trigger
from triggered_vectorizer import triggered_vectorizer

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np


class DOAHeatmap(QtWidgets.QWidget):
    updated = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, azimuths, elevations, num_peaks=3):
        super().__init__()

        self.azimuths = np.asarray(azimuths, dtype=float)
        self.elevations = np.asarray(elevations, dtype=float)
        self.num_peaks = int(max(1, num_peaks))

        self.az_bins = len(self.azimuths)
        self.el_bins = len(self.elevations)
        self.el_min = float(np.min(self.elevations))
        self.el_max = float(np.max(self.elevations))

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.graphics = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics)

        self.plot_item = self.graphics.addPlot(row=0, col=0)
        self.plot_item.setAspectLocked(True)
        self.plot_item.hideAxis('bottom')
        self.plot_item.hideAxis('left')

        self.cmap_item = pg.PColorMeshItem()
        self.plot_item.addItem(self.cmap_item)

        raw_lut = np.asarray(pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256), dtype=float)
        if raw_lut.max() <= 1.0:
            raw_lut *= 255.0
        raw_lut = np.clip(raw_lut, 0, 255).astype(np.uint8)
        if raw_lut.shape[1] == 3:
            raw_lut = np.hstack((raw_lut, np.full((raw_lut.shape[0], 1), 255, dtype=np.uint8)))
        self._lut = [QtGui.QColor(int(r), int(g), int(b), int(a)) for r, g, b, a in raw_lut]
        self.cmap_item.setLookupTable(self._lut)
        self.cmap_item.setLevels((-100.0, 0.0))

        self._build_mesh_geometry()
        self._draw_polar_grid()

        self.peak_markers = []
        self.peak_labels = []
        brushes = [
            pg.mkBrush(255, 0, 0, 220),
            pg.mkBrush(255, 165, 0, 220),
            pg.mkBrush(0, 200, 255, 220),
        ]
        for i in range(self.num_peaks):
            marker = pg.ScatterPlotItem(
                size=max(8, 12 - 2 * i),
                pen=pg.mkPen('w', width=2),
                brush=brushes[i % len(brushes)],
                symbol='o',
            )
            label = pg.TextItem(text="", color=(255, 255, 255), anchor=(0, 1))
            self.plot_item.addItem(marker)
            self.plot_item.addItem(label)
            self.peak_markers.append(marker)
            self.peak_labels.append(label)

        r_outer = self.radius_edges.max()
        pad = 0.08 * r_outer
        self.plot_item.setXRange(-r_outer - pad, r_outer + pad, padding=0)
        self.plot_item.setYRange(-r_outer - pad, r_outer + pad, padding=0)

        self.updated.connect(self.update_image)

    def _centers_to_edges(self, centers, periodic=False, period=None):
        centers = np.asarray(centers, dtype=float)
        edges = np.empty(len(centers) + 1, dtype=float)
        mids = 0.5 * (centers[:-1] + centers[1:])
        edges[1:-1] = mids
        if periodic:
            if period is None:
                raise ValueError("period must be provided for periodic axes")
            step0 = centers[1] - centers[0]
            edges[0] = centers[0] - 0.5 * step0
            edges[-1] = edges[0] + period
        else:
            edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
            edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
        return edges

    def _build_mesh_geometry(self):
        az_edges_deg = self._centers_to_edges(self.azimuths, periodic=True, period=360.0)
        az_edges_rad = np.deg2rad(az_edges_deg)
        el_edges = self._centers_to_edges(self.elevations, periodic=False)
        self.radius_edges = np.maximum(self.el_max - el_edges, 0.0)
        Theta, R = np.meshgrid(az_edges_rad, self.radius_edges)
        self.X = R * np.cos(Theta)
        self.Y = R * np.sin(Theta)

    def _draw_polar_grid(self):
        r_outer = self.radius_edges.max()
        for el in [0, 15, 30, 45, 60, 75, 90]:
            if self.el_min <= el <= self.el_max:
                r = self.el_max - el
                circle = QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
                circle.setPen(pg.mkPen((120, 120, 120, 120), width=1))
                self.plot_item.addItem(circle)
                txt = pg.TextItem(text=f"{el}°", color=(180, 180, 180))
                txt.setPos(r, 0)
                self.plot_item.addItem(txt)

        for az in range(0, 360, 30):
            th = np.deg2rad(az)
            x = r_outer * np.cos(th)
            y = r_outer * np.sin(th)
            self.plot_item.plot([0, x], [0, y], pen=pg.mkPen((120, 120, 120, 120), width=1))
            tx = 1.08 * r_outer * np.cos(th)
            ty = 1.08 * r_outer * np.sin(th)
            txt = pg.TextItem(text=f"{az}°", color=(180, 180, 180), anchor=(0.5, 0.5))
            txt.setPos(tx, ty)
            self.plot_item.addItem(txt)

    def _peak_to_xy(self, az_deg, el_deg):
        r = self.el_max - el_deg
        th = np.deg2rad(az_deg)
        return r * np.cos(th), r * np.sin(th)

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, data):
        expected_len = self.el_bins * self.az_bins
        if data.size != expected_len:
            return

        Z = np.reshape(data, (self.el_bins, self.az_bins))
        self.cmap_item.setData(self.X, self.Y, Z)

        zmin = float(np.nanmin(Z))
        zmax = float(np.nanmax(Z))
        if np.isfinite(zmin) and np.isfinite(zmax):
            if zmax <= zmin:
                zmax = zmin + 1e-6
            self.cmap_item.setLevels((zmin, zmax))

        flat = Z.ravel()
        n_pick = min(self.num_peaks, flat.size)
        if n_pick <= 0:
            return

        peak_idx = np.argpartition(flat, -n_pick)[-n_pick:]
        peak_idx = peak_idx[np.argsort(flat[peak_idx])[::-1]]

        label_d = 0.03 * max(self.radius_edges.max(), 1.0)
        for rank in range(self.num_peaks):
            if rank < n_pick:
                idx = int(peak_idx[rank])
                el_idx, az_idx = np.unravel_index(idx, Z.shape)
                peak_az = float(self.azimuths[az_idx])
                peak_el = float(self.elevations[el_idx])
                peak_val = float(Z[el_idx, az_idx])
                x, y = self._peak_to_xy(peak_az, peak_el)
                self.peak_markers[rank].setData([x], [y])
                self.peak_labels[rank].setPos(x + label_d, y + label_d)
                self.peak_labels[rank].setText(f"#{rank+1} Az {peak_az:.1f}°\nEl {peak_el:.1f}°\n{peak_val:.1f} dB")
            else:
                self.peak_markers[rank].setData([], [])
                self.peak_labels[rank].setText("")


class VectorSinkCallback(gr.sync_block):
    def __init__(self, vector_len):
        gr.sync_block.__init__(
            self,
            name="vector_sink_callback",
            in_sig=[(np.float32, vector_len)],
            out_sig=[]
        )
        self._lock = threading.Lock()
        self._latest = None

    def work(self, input_items, output_items):
        for vec in input_items[0]:
            arr = np.array(vec, copy=True)
            with self._lock:
                self._latest = arr
        return len(input_items[0])

    def pop_latest(self):
        with self._lock:
            out = self._latest
            self._latest = None
            return out


class doa_2d(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "Triggered 2D MUSIC DOA Heatmap", catch_exceptions=True)

        self.samp_rate = 2_400_000
        self.center_freq = 903.0e6
        self.signal_offset_hz = 0.0
        self.decimation = 4
        self.post_rate = self.samp_rate / self.decimation
        self.num_elements = 5

        # LoRa parameters for the trigger path
        self.lora_bw = 250_000.0
        self.lora_sf = 7
        self.symbol_time = (2 ** self.lora_sf) / self.lora_bw
        self.samples_per_symbol = int(round(self.post_rate * self.symbol_time))

        # DOA window: 6 LoRa symbols captured after trigger
        self.window_symbols = 6
        self.capture_len = self.window_symbols * self.samples_per_symbol
        self.capture_delay = 0
        self.trigger_rearm_samples = 4 * self.capture_len

        self.kraken_source = krakensdr.krakensdr_source(
            '127.0.0.1', 5000, 5001, self.num_elements, 416.588,
            [40.2] * self.num_elements,
            False
        )

        self.lpf_cutoff = 150e3
        self.lpf_transition = 50e3
        self.channel_taps = firdes.low_pass(
            1.0,
            self.samp_rate,
            self.lpf_cutoff,
            self.lpf_transition,
            fft.window.WIN_HAMMING
        )

        self.channel_filters = []
        for _ in range(self.num_elements):
            xf = filter.freq_xlating_fir_filter_ccf(
                self.decimation,
                self.channel_taps,
                self.signal_offset_hz,
                self.samp_rate
            )
            self.channel_filters.append(xf)

        # Trigger from the first filtered channel only.
        self.lora_trigger = lora_preamble_trigger(
            samp_rate=self.post_rate,
            bandwidth=self.lora_bw,
            spreading_factor=self.lora_sf,
            min_repeats=6,
            bin_tolerance=1,
            peak_ratio_threshold=4.0,
            trigger_advance_symbols=5,
            refractory_symbols=12,
        )

        self.capture = triggered_vectorizer(
            num_channels=self.num_elements,
            window_len=self.capture_len,
            delay_after_trigger=self.capture_delay,
            trigger_threshold=0.5,
            rearm_samples=self.trigger_rearm_samples,
        )

        self.music_block = doa_music.doa_music(
            cpi_size=self.capture_len,
            freq=903.0,
            array_dist=0.1413,
            num_elements=self.num_elements,
            array_type='UCA',
            azimuth_min=0.0,
            azimuth_max=359.0,
            azimuth_step_deg=10.0,
            elevation_min=0.0,
            elevation_max=90.0,
            elevation_step_deg=10.0,
            signal_dimension=3,
            diag_loading=0.0,
            l_elements_x=3,
            l_elements_y=3,
        )

        self.heatmap_widget = DOAHeatmap(
            azimuths=self.music_block.azimuths,
            elevations=self.music_block.elevations,
            num_peaks=3,
        )

        vector_len = len(self.music_block.azimuths) * len(self.music_block.elevations)
        self.vector_sink = VectorSinkCallback(vector_len)

        self.gui_fps = 15
        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self._update_gui_from_sink)
        self.gui_timer.start(int(1000 / self.gui_fps))

        for i in range(self.num_elements):
            self.connect((self.kraken_source, i), (self.channel_filters[i], 0))
            self.connect((self.channel_filters[i], 0), (self.capture, i))

        self.connect((self.channel_filters[0], 0), (self.lora_trigger, 0))
        self.connect((self.lora_trigger, 0), (self.capture, self.num_elements))

        for i in range(self.num_elements):
            self.connect((self.capture, i), (self.music_block, i))

        self.connect((self.music_block, 0), (self.vector_sink, 0))

    def _update_gui_from_sink(self):
        vec = self.vector_sink.pop_latest()
        if vec is not None:
            self.heatmap_widget.update_image(vec)


def main():
    qapp = QtWidgets.QApplication(sys.argv)
    tb = doa_2d()
    tb.heatmap_widget.show()
    tb.start()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    qapp.exec_()


if __name__ == '__main__':
    main()
