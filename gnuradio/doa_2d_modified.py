#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gnuradio import blocks
from gnuradio import gr
from gnuradio import filter
from gnuradio.filter import firdes
import sys
import signal

from gnuradio import krakensdr
import doa_music

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np


class DOAHeatmap(QtWidgets.QWidget):
    updated = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, azimuths, elevations):
        super().__init__()

        self.azimuths = np.asarray(azimuths, dtype=float)
        self.elevations = np.asarray(elevations, dtype=float)

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

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.cmap_item)
        self.graphics.addItem(self.hist, row=0, col=1)

        self._build_mesh_geometry()
        self._draw_polar_grid()

        self.peak_marker = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen('w', width=2),
            brush=pg.mkBrush(255, 0, 0, 220),
            symbol='o'
        )
        self.plot_item.addItem(self.peak_marker)

        self.peak_label = pg.TextItem(
            text="",
            color=(255, 255, 255),
            anchor=(0, 1)
        )
        self.plot_item.addItem(self.peak_label)

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

        self.radius_edges = self.el_max - el_edges
        self.radius_edges = np.maximum(self.radius_edges, 0.0)

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

        peak_flat_idx = np.argmax(Z)
        el_idx, az_idx = np.unravel_index(peak_flat_idx, Z.shape)

        peak_az = float(self.azimuths[az_idx])
        peak_el = float(self.elevations[el_idx])
        peak_val = float(Z[el_idx, az_idx])

        x, y = self._peak_to_xy(peak_az, peak_el)
        self.peak_marker.setData([x], [y])

        label_d = 0.03 * max(self.radius_edges.max(), 1.0)
        self.peak_label.setPos(x + label_d, y + label_d)
        self.peak_label.setText(f"Az {peak_az:.1f}°\nEl {peak_el:.1f}°\n{peak_val:.1f} dB")


class VectorSinkCallback(gr.sync_block):
    def __init__(self, heatmap_widget, vector_len):
        gr.sync_block.__init__(
            self,
            name="vector_sink_callback",
            in_sig=[(np.float32, vector_len)],
            out_sig=[]
        )
        self.heatmap_widget = heatmap_widget

    def work(self, input_items, output_items):
        for vec in input_items[0]:
            self.heatmap_widget.updated.emit(vec)
        return len(input_items[0])


class doa_2d(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "2D MUSIC DOA Heatmap", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = 2_400_000
        self.center_freq = 903.0e6
        self.signal_offset_hz = 0.0          # set this if the LoRa signal is offset from DC
        self.decimation = 4
        self.post_rate = self.samp_rate / self.decimation

        self.cpi_size_raw = 2**20
        self.cpi_size = self.cpi_size_raw // self.decimation
        self.num_elements = 5

        ##################################################
        # Source
        ##################################################
        self.kraken_source = krakensdr.krakensdr_source(
            '127.0.0.1', 5000, 5001, self.num_elements, 416.588,
            [40.2] * self.num_elements,
            False
        )

        ##################################################
        # Channel-select / decimation filter
        ##################################################
        # For a 250 kHz LoRa signal:
        # - passband should safely include the occupied bandwidth
        # - output rate after decim=4 is 600 kS/s
        # - Nyquist at output is 300 kHz
        #
        # These are good starting values:
        self.lpf_cutoff = 150e3
        self.lpf_transition = 50e3

        self.channel_taps = firdes.low_pass(
            1.0,
            self.samp_rate,
            self.lpf_cutoff,
            self.lpf_transition,
            firdes.WIN_HAMMING
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

        ##################################################
        # Stream-to-vector
        ##################################################
        self.stream_to_vector_blocks = []
        for _ in range(self.num_elements):
            b = blocks.stream_to_vector(gr.sizeof_gr_complex, self.cpi_size)
            self.stream_to_vector_blocks.append(b)

        ##################################################
        # MUSIC block
        ##################################################
        self.music_block = doa_music.doa_music(
            cpi_size=self.cpi_size,          # already-decimated vector length
            freq=903.0,                      # MHz
            array_dist=0.1413,
            num_elements=self.num_elements,
            array_type='UCA',
            azimuth_min=0.0,
            azimuth_max=359.0,
            azimuth_step_deg=1.0,
            elevation_min=0.0,
            elevation_max=90.0,
            elevation_step_deg=1.0,
            signal_dimension=1,
            diag_loading=1e-3
        )

        ##################################################
        # Heatmap + sink
        ##################################################
        self.heatmap_widget = DOAHeatmap(
            azimuths=self.music_block.azimuths,
            elevations=self.music_block.elevations
        )

        vector_len = len(self.music_block.azimuths) * len(self.music_block.elevations)
        self.vector_sink = VectorSinkCallback(self.heatmap_widget, vector_len)

        ##################################################
        # Connections
        ##################################################
        for i in range(self.num_elements):
            self.connect((self.kraken_source, i), (self.channel_filters[i], 0))
            self.connect((self.channel_filters[i], 0), (self.stream_to_vector_blocks[i], 0))
            self.connect((self.stream_to_vector_blocks[i], 0), (self.music_block, i))

        self.connect((self.music_block, 0), (self.vector_sink, 0))


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