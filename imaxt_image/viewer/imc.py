import numpy as np
import xarray as xr
import holoviews as hv
import datashader as ds
import pandas as pd
import os

from holoviews.operation.datashader import regrid
from datashader.utils import ngjit
import cv2
import scipy


class IMCViewer:
    def __init__(
        self,
        IMC_name,
        ROI_number,
        path=None,
        use_panel_file=False,
        panel_file=None,
        panel_skip_rows=4,
    ):
        self.IMC_name = IMC_name
        self.ROI_number = ROI_number
        if path:
            self.filepath = path
        else:
            self.filepath = "/data/meds1_d/storage/raw/imc/"

        print("Loading IMC dataset")
        path_to_open = (
            os.path.join(self.filepath, self.IMC_name) + "/Q00" + str(self.ROI_number)
        )
        ds = xr.open_zarr(path_to_open)
        image = ds["Q00" + str(ROI_number)]
        image.load()
        channels = pd.DataFrame(image.meta[0]["q_channels"])
        self.data = image
        self.channels = channels
        self.channels["good"] = np.zeros(len(self.channels))

        if use_panel_file:
            panel = pd.read_excel(panel_file, skiprows=panel_skip_rows)
            for item in self.channels.metal:
                if "(" in item:
                    metal = item[:2]
                    number = item[3:6]
                    tomatch = number + metal
                    match = panel[panel["Metal"] == tomatch].Target.tolist()
                    if len(match) > 0:
                        self.channels.loc[self.channels.metal == item, "target"] = match
                        self.channels.loc[self.channels.metal == item, "good"] = 1
        else:
            self.channels["good"] = np.ones(len(self.channels))

    def prepare_data(self, ch_to_display, cropped=False):
        data = []
        mins = []
        maxs = []
        colors = []

        if cropped:
            for ch in ch_to_display:
                ch_index = self.channels[
                    self.channels["target"] == ch[0]
                ].index.tolist()[0]
                data_ch = self.data_cropped[ch_index, :, :]
                if ch[4]:
                    data_ch.data = scipy.signal.medfilt2d(data_ch.data, ch[4])
                    # data_ch.data = cv2.medianBlur(data_ch.data, ch[4])
                data.append(data_ch)
                mins.append(ch[1])
                maxs.append(ch[2])
                colors.append(ch[3])
            self.selected_data_cropped = data
            self.mins_cropped = mins
            self.maxs_cropped = maxs
            self.colors_cropped = colors
        else:
            for ch in ch_to_display:
                ch_index = self.channels[
                    self.channels["target"] == ch[0]
                ].index.tolist()[0]
                data_ch = self.data[ch_index, :, :]
                if ch[4]:
                    data_ch.data = scipy.signal.medfilt2d(data_ch.data, ch[4])
                    # data_ch.data = cv2.medianBlur(data_ch.data, ch[4])
                data.append(data_ch)
                mins.append(ch[1])
                maxs.append(ch[2])
                colors.append(ch[3])
            self.selected_data = data
            self.mins = mins
            self.maxs = maxs
            self.colors = colors

    def prepare_texts(self, ch_to_display):
        texts = []
        for n in range(len(ch_to_display)):
            tx = hv.Points((0, 0), label=ch_to_display[n][5]).opts(
                color=ch_to_display[n][3]
            )
            texts.append(tx)
            self.captions_legend = texts

    def select_color_rgb(self, colour):
        if colour == "red":
            return 1, 0, 0
        elif colour == "green":
            return 0, 1, 0
        elif colour == "blue":
            return 0, 0, 1
        elif colour == "cyan":
            return 0, 1, 1
        elif colour == "yellow":
            return 1, 1, 0
        elif colour == "magenta":
            return 1, 0, 1
        elif colour == "grey":
            return 1, 1, 1

    def colorize_channel_whitefield(self, image, factors):

        rscale, gscale, bscale = factors
        r = image / image.max()
        g = image / image.max()
        b = image / image.max()
        rinv = abs(1 - r * rscale)
        ginv = abs(1 - g * gscale)
        binv = abs(1 - b * bscale)

        rgb2 = np.ones((image.shape[0], image.shape[1], 3))
        rgb2[:, :, 0] = rinv
        rgb2[:, :, 1] = ginv
        rgb2[:, :, 2] = binv

        return rgb2

    def combine_ch_colors(self):
        channels = self.selected_data
        min_values = self.mins
        max_values = self.maxs
        colours = self.colors
        xs, ys = channels[0]["x"], channels[0]["y"]

        color_arrays = []
        for col in colours:
            arr = self.select_color_rgb(col)
            color_arrays.append(arr)

        rgb_tot = np.zeros((channels[0].shape[0], channels[0].shape[1], 3))

        for n in range(len(channels)):
            ch_data = channels[n].data
            if "p" in str(max_values[n]):
                percentile = float(max_values[n][1:])
                threshold = np.percentile(ch_data, percentile)
                print(
                    "Auto Max value for channel: " + str(n) + " is: " + str(threshold)
                )
                if threshold == 0:
                    threshold = 0.01
            else:
                threshold = max_values[n]
            if "p" in str(min_values[n]):
                percentile_min = float(min_values[n][1:])
                threshold_min = np.percentile(ch_data, percentile_min)
                print(
                    "Auto Min value for channel: "
                    + str(n)
                    + " is: "
                    + str(threshold_min)
                )
                if threshold_min == 0:
                    threshold_min = 0.01
            else:
                threshold_min = min_values[n]
            ch = ds.utils.orient_array(channels[n])
            out = norm_colorize(ch, threshold_min, threshold, color_arrays[n], rgb_tot)
            rgb_tot = out

        self.rendered_image = hv.RGB(
            (xs, ys[::-1], rgb_tot[:, :, 0], rgb_tot[:, :, 1], rgb_tot[:, :, 2])
        )
        self.rgb_tot = rgb_tot

    def combine_ch_colors_cropped(self):
        channels = self.selected_data_cropped
        min_values = self.mins_cropped
        max_values = self.maxs_cropped
        colours = self.colors_cropped
        xs, ys = channels[0]["x"], channels[0]["y"]

        color_arrays = []
        for col in colours:
            arr = self.select_color_rgb(col)
            color_arrays.append(arr)

        rgb_tot = np.zeros((channels[0].shape[0], channels[0].shape[1], 3))

        for n in range(len(channels)):
            ch_data = channels[n].data
            if "p" in str(max_values[n]):
                percentile = float(max_values[n][1:])
                threshold = np.percentile(ch_data, percentile)
                print(
                    "Auto Max value for channel: " + str(n) + " is: " + str(threshold)
                )
                if threshold == 0:
                    threshold = 0.01
            else:
                threshold = max_values[n]
            if "p" in str(min_values[n]):
                percentile_min = float(min_values[n][1:])
                threshold_min = np.percentile(ch_data, percentile_min)
                print(
                    "Auto Min value for channel: "
                    + str(n)
                    + " is: "
                    + str(threshold_min)
                )
                if threshold_min == 0:
                    threshold_min = 0.01
            else:
                threshold_min = min_values[n]
            ch = ds.utils.orient_array(channels[n])
            out = norm_colorize(ch, threshold_min, threshold, color_arrays[n], rgb_tot)
            rgb_tot = out

        self.rendered_image_cropped = hv.RGB(
            (xs, ys[::-1], rgb_tot[:, :, 0], rgb_tot[:, :, 1], rgb_tot[:, :, 2])
        )
        self.rgb_tot_cropped = rgb_tot

    def scalebar(self, x_range, y_range):
        if x_range and y_range:
            x0, x1 = x_range
            y0, y1 = y_range
            x_span = x1 - x0
            y_span = y1 - y0
            y0 = y0 + int(y_span / 20)
            y1 = y0 + int(y_span / 50)
            ytext = y1 + int(y_span / 30)
            x0 = x1 - self.scalebar_size - int(x_span / 30)
            x1 = x1 - int(x_span / 30)
            rect = hv.Rectangles((x0, y0, x1, y1)).opts(color="w")
            text = hv.Text(x0, ytext, str(self.scalebar_size) + " um").opts(
                text_color="w", text_align="left", text_font="Helvetica"
            )
            scalebar = rect * text
            return scalebar
        else:
            x0 = 0
            y0 = 0
            x1 = self.rendered_image.data["x"].size
            y1 = self.rendered_image.data["y"].size
            x_span = x1 - x0
            y_span = y1 - y0
            y0 = y0 + int(y_span / 20)
            y1 = y0 + int(y_span / 50)
            x0 = x1 - self.scalebar_size - int(x_span / 40)
            x1 = x1 - int(x_span / 30)
            ytext = y1 + int(y_span / 30)
            rect = hv.Rectangles((x0, y0, x1, y1)).opts(color="w")
            text = hv.Text(x0, ytext, str(self.scalebar_size) + " um").opts(
                text_color="w", text_align="left", text_font="Helvetica"
            )
            scalebar = rect * text
            return scalebar

    def scalebar_cropped(self, x_range, y_range):
        if x_range and y_range:
            x0, x1 = x_range
            y0, y1 = y_range
            x_span = x1 - x0
            y_span = y1 - y0
            y0 = y0 + int(y_span / 20)
            y1 = y0 + int(y_span / 50)
            ytext = y1 + int(y_span / 30)
            x0 = x1 - self.scalebar_size_cropped - int(x_span / 30)
            x1 = x1 - int(x_span / 30)
            rect = hv.Rectangles((x0, y0, x1, y1)).opts(color="w")
            text = hv.Text(x0, ytext, str(self.scalebar_size_cropped) + " um").opts(
                text_color="w", text_align="left", text_font="Helvetica"
            )
            scalebar = rect * text
            return scalebar
        else:
            x0 = self.x_range[0]
            y0 = self.y_range[0]
            x1 = self.x_range[1]
            y1 = self.y_range[1]
            x_span = x1 - x0
            y_span = y1 - y0
            y0 = y0 + int(y_span / 20)
            y1 = y0 + int(y_span / 50)
            x0 = x1 - self.scalebar_size_cropped - int(x_span / 40)
            x1 = x1 - int(x_span / 30)
            ytext = y1 + int(y_span / 30)
            rect = hv.Rectangles((x0, y0, x1, y1)).opts(color="w")
            text = hv.Text(x0, ytext, str(self.scalebar_size_cropped) + " um").opts(
                text_color="w", text_align="left", text_font="Helvetica"
            )
            scalebar = rect * text
            return scalebar

    def print_channels(self):
        print(list(self.channels.target))

    def view(
        self,
        ch_to_display,
        scalebar_size,
        rescale_factor,
        show_legend=True,
        legend_position="bottom_left",
    ):

        w = self.data.sizes["x"]
        h = self.data.sizes["y"]

        self.prepare_data(ch_to_display)
        self.combine_ch_colors()
        self.prepare_texts(ch_to_display)

        self.rangexy = hv.streams.RangeXY(source=self.rendered_image)
        self.scalebar_size = scalebar_size

        self.regridded_image = (
            regrid(self.rendered_image, streams=[self.rangexy])
            .options(framewise=True)
            .opts(width=int(w / rescale_factor), height=int(h / rescale_factor))
        )
        self.scalebar_image = self.regridded_image * hv.DynamicMap(
            self.scalebar, streams=[self.rangexy]
        )

        self.overlay_image = hv.Overlay([self.scalebar_image] + self.captions_legend)
        self.final_image = self.overlay_image.collate().opts(
            show_legend=show_legend, legend_position=legend_position
        )
        return self.final_image

    def select_zoomed_ROI(self):
        self.ROI_final = self.final_image.select(
            x=self.rangexy.x_range, y=self.rangexy.y_range
        )
        self.ROI_image = self.rendered_image.select(
            x=self.rangexy.x_range, y=self.rangexy.y_range
        )
        self.x_range = self.rangexy.x_range
        self.y_range = self.rangexy.y_range
        print("ROI Selected")
        print("X_range = " + str(self.x_range))
        print("Y_range = " + str(self.y_range))

    def crop_dataset(self):
        if self.x_range and self.y_range:
            self.data_cropped = self.data.sel(
                x=range(int(self.x_range[0]), int(self.x_range[1])),
                y=range(int(self.y_range[0]), int(self.y_range[1])),
            )
        else:
            print("run select_zoomed_roi first")

    def save_image(
        self,
        filename,
        individual=False,
        scalebar=True,
        scalebar_size=100,
        return_rgb=False,
        downsample=None,
    ):

        if individual:
            r = self.ROI_image_individual.data["R"]
            g = self.ROI_image_individual.data["G"]
            b = self.ROI_image_individual.data["B"]
        else:
            r = self.ROI_image.data["R"]
            g = self.ROI_image.data["G"]
            b = self.ROI_image.data["B"]
        r = r / r.max() * 255
        g = g / g.max() * 255
        b = b / b.max() * 255
        rgb = np.zeros((r.shape[0], r.shape[1], 3))
        rgb[:, :, 0] = b
        rgb[:, :, 1] = g
        rgb[:, :, 2] = r
        rgb = cv2.flip(rgb, 0)

        if scalebar:

            h = rgb.shape[0]
            w = rgb.shape[1]

            start = (w - scalebar_size - int(w / 20), h - 10 - int(h / 20))
            end = (w - int(w / 20), h - int(h / 20))
            start_text = (w - scalebar_size - int(w / 20), h - 10 - int(h / 20) - 10)
            text = str(scalebar_size) + " um"

            rgb2 = cv2.rectangle(rgb, start, end, (255, 255, 255), -1)
            rgb3 = cv2.putText(
                rgb2,
                text,
                start_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if downsample:
                width = int(rgb.shape[1] * downsample)
                height = int(rgb.shape[0] * downsample)
                dim = (width, height)
                rgb4 = cv2.resize(rgb3, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(filename + ".png", rgb4)
            else:
                cv2.imwrite(filename + ".png", rgb3)

        else:
            if downsample:
                width = int(rgb.shape[1] * downsample)
                height = int(rgb.shape[0] * downsample)
                dim = (width, height)
                rgb2 = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(filename + ".png", rgb2)
            else:
                cv2.imwrite(filename + ".png", rgb)

        if return_rgb:
            return rgb

    def print_individual_images(
        self,
        channels_to_show,
        scalebar_size,
        colour,
        show_legend=False,
        legend_position="bottom_left",
        print_histogram=False,
        hist_range=(0, 255),
        hist_colour="blue",
        path=".",
        median=None,
        print_nuclear_ch=False,
        nuclear_channel=None,
        nuclear_max="p99",
        nuclear_colour="blue",
        downsample=None,
    ):

        self.individual_images = list()

        for item in channels_to_show:
            channel = item[0]
            max_value = item[1]
            if print_nuclear_ch:
                ch_to_display = [
                    [channel, 0, max_value, colour, median, channel],
                    [
                        nuclear_channel,
                        0,
                        nuclear_max,
                        nuclear_colour,
                        median,
                        nuclear_channel,
                    ],
                ]
            else:
                ch_to_display = [[channel, 0, max_value, colour, median, channel]]

            self.prepare_data(ch_to_display)
            self.combine_ch_colors()
            self.ROI_image_individual = self.rendered_image.select(
                x=self.x_range, y=self.y_range
            )
            self.individual_images.append(self.ROI_image_individual)
            rgb = self.save_image(
                os.path.join(path, channel),
                individual=True,
                scalebar=True,
                scalebar_size=scalebar_size,
                return_rgb=True,
                downsample=downsample,
            )

            if print_histogram:
                hist_data = np.histogram(rgb[:, :, 1], bins=10, range=hist_range)
                hist = hv.Histogram(hist_data).opts(
                    color=hist_colour, alpha=0.4, toolbar=None, xlabel="Intensity"
                )

                hist_intensities = hist_data[0]
                hist_borders = hist_data[1]

                with open(os.path.join(path, channel + "_hist.png"), "w") as outfile:
                    for n in range(len(hist_intensities)):
                        outfile.write("\n")
                        outfile.write(
                            str(hist_borders[n]) + "-" + str(hist_borders[n + 1]) + ","
                        )
                        outfile.write(str(hist_intensities[n]))

                hv.save(hist, os.path.join(path, channel + "_hist.png"))

    def print_all_channels(
        self,
        max_value,
        min_value,
        scalebar_size,
        colour,
        show_legend=False,
        legend_position="bottom_left",
        print_histogram=False,
        hist_range=(0, 255),
        hist_colour="blue",
        path=".",
        median=None,
        print_nuclear_ch=False,
        nuclear_channel=None,
        nuclear_max="p99",
        nuclear_colour="blue",
        downsample=None,
    ):

        self.all_images = list()

        for index, row in self.channels.iterrows():
            if row["good"] == 1:
                channel = row["target"]
                if print_nuclear_ch:
                    ch_to_display = [
                        [channel, 0, max_value, colour, median, channel],
                        [
                            nuclear_channel,
                            0,
                            nuclear_max,
                            nuclear_colour,
                            median,
                            channel,
                        ],
                    ]
                else:
                    ch_to_display = [[channel, 0, max_value, colour, median, channel]]
                self.prepare_data(ch_to_display)
                self.combine_ch_colors()
                self.ROI_image_individual = self.rendered_image.select(
                    x=self.x_range, y=self.y_range
                )
                self.all_images.append(self.ROI_image_individual)
                rgb = self.save_image(
                    os.path.join(path, channel),
                    individual=True,
                    scalebar=True,
                    scalebar_size=scalebar_size,
                    return_rgb=True,
                    downsample=downsample,
                )

                if print_histogram:
                    hist_data = np.histogram(rgb[:, :, 1], bins=10, range=hist_range)
                    hist = hv.Histogram(hist_data).opts(
                        color=hist_colour, alpha=0.4, toolbar=None, xlabel="Intensity"
                    )
                    hist_intensities = hist_data[0]
                    hist_borders = hist_data[1]

                    with open(
                        os.path.join(path, channel + "_hist.png"), "w"
                    ) as outfile:
                        for n in range(len(hist_intensities)):
                            outfile.write("\n")
                            outfile.write(
                                str(hist_borders[n])
                                + "-"
                                + str(hist_borders[n + 1])
                                + ","
                            )
                            outfile.write(str(hist_intensities[n]))

                    hv.save(hist, os.path.join(path, channel + "_hist.png"))


@ngjit
def norm_colorize(agg, min_val, max_val, colour, tot):
    out = np.zeros((agg.shape[0], agg.shape[1], 3), dtype=np.float32)
    range_val = max_val - min_val
    col, rows = agg.shape
    for x in range(col):
        for y in range(rows):
            val = agg[x, y]
            norm = (val - min_val) / range_val
            if norm > 1:
                norm = 1

            out[x, y, 0] = tot[x, y, 0] + norm * colour[0]
            out[x, y, 1] = tot[x, y, 1] + norm * colour[1]
            out[x, y, 2] = tot[x, y, 2] + norm * colour[2]

    return out
