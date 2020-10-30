import asyncio
import os
from contextlib import suppress
from glob import glob
from math import ceil
from pathlib import Path

import dask
import datashader as ds
import holoviews as hv
import holoviews.operation.datashader as hd
import numpy as np
import panel as pn
import xarray as xr
from astropy.visualization import PercentileInterval
from bokeh.models import HoverTool
from bokeh.util.serialization import make_globally_unique_id
from holoviews import streams
from holoviews.plotting.links import RangeToolLink


css = """
.custom-wbox > div.bk {
    padding-right: 10px;
}
.scrollable {
    overflow: auto !important;
}
"""

js_files = {
    "jquery": "https://code.jquery.com/jquery-1.11.1.min.js",
    "goldenlayout": "https://golden-layout.com/files/latest/js/goldenlayout.min.js",
}
css_files = [
    "https://golden-layout.com/files/latest/css/goldenlayout-base.css",
    "https://golden-layout.com/files/latest/css/goldenlayout-dark-theme.css",
]

template = """
{%% extends base %%}
<!-- goes in body -->
{%% block contents %%}
{%% set context = '%s' %%}
{%% if context == 'notebook' %%}
    {%% set slicer_id = get_id() %%}
    <div id='{{slicer_id}}'></div>
{%% endif %%}

<script>
var config = {
    settings: {
        hasHeaders: true,
        constrainDragToContainer: true,
        reorderEnabled: true,
        selectionEnabled: false,
        popoutWholeStack: false,
        blockedPopoutsThrowError: true,
        closePopoutsOnUnload: true,
        showPopoutIcon: false,
        showMaximiseIcon: true,
        showCloseIcon: false
    },
    content: [{
        type: 'row',
        content:[
            {
                type: 'component',
                componentName: 'view',
                componentState: { model: '{{ embed(roots.C) }}',
                                  title: 'Controls',
                                  width: 250,
                                  css_classes:['scrollable']},
                isClosable: false,
            },
            {
                type: 'column',
                content: [
                    {
                        type: 'row',
                        content:[
                            {
                                type: 'component',
                                componentName: 'view',
                                componentState: { model: '{{ embed(roots.A) }}', title: 'View', color: '#15191C'},
                                isClosable: false,
                            }
                        ]
                    }
                ]
            }
        ]
    }]
};

{%% if context == 'notebook' %%}
    var myLayout = new GoldenLayout( config, '#' + '{{slicer_id}}' );
    $('#' + '{{slicer_id}}').css({width: '100%%', height: '{{height}}', margin: '0px'})
{%% else %%}
    var myLayout = new GoldenLayout( config );
{%% endif %%}

myLayout.registerComponent('view', function( container, componentState ){
    const {width, css_classes} = componentState
    if(width)
      container.on('open', () => container.setSize(width, container.height))
    if (css_classes)
      css_classes.map((item) => container.getElement().addClass(item))
    container.setTitle(componentState.title)
    container.getElement().html(componentState.model);
    container.getElement().css( 'background-color', componentState.color );
    container.on('resize', () => window.dispatchEvent(new Event('resize')))
});


myLayout.init();
</script>
{%% endblock %%}
"""


def setup_notebook():
    pn.extension(js_files=js_files, raw_css=[css], css_files=css_files)
    hv.renderer("bokeh").theme = "dark_minimal"


def remove_bokeh_logo(plot, element):
    plot.state.toolbar.logo = None


async def async_play(dv, n1, n2, step, wait, saveas):
    if saveas is not None:
        for f in glob("tmpplot_???.png"):
            os.unlink(f)

    for i in range(n1, n2, step):
        dv.buffer = False
        dv.goto(i)
        dv.buffer = True
        if saveas is not None:
            element = dv.main.data[()]
            p = hv.render(element, backend="matplotlib")
            p.axes[0].set_axis_off()
            p.savefig(
                f"tmpplot_{i-n1:03d}.png", bbox_inches="tight", pad_inches=0, dpi=300
            )
        await asyncio.sleep(wait)
    os.system(f"ffmpeg -y -r 1 -i tmpplot_%03d.png -vcodec mpeg4 {saveas}")


class StptDataset:
    def __init__(self, sample, path=None):
        self.path = Path(path)

        mos = f"{self.path / sample}/mos.zarr"
        ds = xr.open_zarr(mos).sel(type="mosaic")
        levels = ds.attrs["multiscale"]["datasets"]
        self.ds = {
            k["level"]: xr.open_zarr(mos, group=k["path"]).sel(type="mosaic")
            for k in levels
        }
        self.ds = {k: self.ds[k] * self.bscale + self.bzero for k in self.ds}
        self.scl = [k["level"] for k in levels]
        self.nlevels = len(self.scl)

    def __repr__(self):
        return self.ds[1].__repr__()

    def __len__(self):
        return len(self.ds[1])

    @property
    def channels(self):
        return list(self.ds[1].channel.values)

    @property
    def shape(self):
        xs = len(self.ds[1].x)
        ys = len(self.ds[1].y)
        return (ys, xs)

    @property
    def nz(self):
        return len(self.ds[1].z)

    @property
    def nslides(self):
        return self.nz * len(self.ds[1])

    @property
    def bscale(self):
        return self.ds[1].attrs.get("bscale", 0.001)

    @property
    def bzero(self):
        return self.ds[1].attrs.get("bzero", -10)

    def __getitem__(self, key):
        return self.ds[key]

    def compute_scale(self, x_range, y_range, res=900):
        if not x_range:
            return self.scl[-1]

        xdiff = abs(x_range[0] - x_range[1])
        scale = np.array([0, res * 1, res * 2, res * 4, res * 8, res * 16, res * 32])
        scl = scale.searchsorted(xdiff)
        scl = min(self.nlevels, scl)
        return self.scl[scl - 1]

    def display_interval(self, channel, percentile):
        zs = PercentileInterval(percentile)
        data = self.ds[8]["S001"].sel(channel=channel, z=0).values
        mask = data > 0
        return zs.get_limits(data[mask])

    def info(self):
        print(f"Channels: {self.channels}")
        print(f"Physical Slices: {len(self.ds[1])}")
        print(f"Optical Slices: {self.nz}")
        print(f"Mosaic size: {self.shape}")


class regrid(hd.regrid):
    def _process(self, element, key=None):

        # Compute coords, anges and size
        x, y = element.kdims
        coords = tuple(element.dimension_values(d, expanded=False) for d in [x, y])
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info

        # This is how to make panning smoother
        # https://github.com/holoviz/holoviews/issues/4435
        xdiff = (x_range[1] - x_range[0]) / 2
        ydiff = (y_range[1] - y_range[0]) / 2

        x_range = (max(0, x_range[0] - xdiff), min(x_range[1] + xdiff, coords[0].max()))
        y_range = (max(0, y_range[0] - ydiff), min(y_range[1] + ydiff, coords[1].max()))

        # (xstart, xend), (ystart, yend) = (x_range, y_range)
        # xspan, yspan = (xend - xstart), (yend - ystart)
        interp = self.p.interpolation or None
        if interp == "bilinear":
            interp = "linear"

        # Compute bounds (converting datetimes)
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(
            x_range, y_range, xs, ys, xtype, ytype
        )

        params = dict(bounds=(x0, y0, x1, y1))
        if width == 0 or height == 0:
            if width == 0:
                params["xdensity"] = 1
            if height == 0:
                params["ydensity"] = 1
            return element.clone((xs, ys, np.zeros((height, width))), **params)

        cvs = ds.Canvas(
            plot_width=width, plot_height=height, x_range=x_range, y_range=y_range
        )

        # Apply regridding to each value dimension
        regridded = {}
        arrays = self._get_xarrays(element, coords, xtype, ytype)
        agg_fn = self._get_aggregator(element, add_field=False)
        for vd, xarr in arrays.items():
            rarray = cvs.raster(xarr, upsample_method=interp, downsample_method=agg_fn)

            regridded[vd] = rarray
        regridded = xr.Dataset(regridded)

        return element.clone(
            regridded, datatype=["xarray"] + element.datatype, **params
        )


class StptDataViewer:
    def __init__(self, name, path="/data/meds1_b/processed/STPT/"):
        self.name = name
        self.ds = StptDataset(name, path=path)
        self.dataset = self.ds
        self.buffer = True

    # Setup GUI

    def setup_template(self, height=600):
        self.tmpl = pn.Template(
            template=(template % "server"), nb_template=(template % "notebook")
        )
        self.tmpl.nb_template.globals["get_id"] = make_globally_unique_id
        self.tmpl.add_variable("height", f"{height}px")

    def setup_streams(self):
        self.range_xy = streams.RangeXY()
        self.pipe = streams.Pipe(data=[])
        self.pointer = streams.PointerXY()

    def setup_controller(self, channels=None):
        slider = pn.widgets.IntSlider(
            name="Slide No.", start=1, end=self.ds.nslides, value=1, width=200
        )
        slider.param.watch(self.update_slide, "value_throttled")

        self.controller = pn.WidgetBox(
            slider,
            css_classes=["widget-box", "custom-wbox"],
            sizing_mode="stretch_both",
        )

    # Play methods

    def play(self, start=None, end=None, step=1, wait=2, saveas=None):
        start = start or 1
        end = end or self.ds.nslides
        if end > self.ds.nslides:
            end = self.ds.nslides
        if end < 1:
            end = 1
        self.coro = asyncio.create_task(
            async_play(self, start, end, step, wait, saveas)
        )
        print(f"The display will now cycle from slices {start} to {end}")
        if saveas is not None:
            print(f"At the end, a movie file will ba saved as {saveas}")

    def stop(self):
        with suppress(Exception):
            self.coro.cancel()

    def goto(self, section):
        self.controller[0].value = section
        self.pipe.send({"section": section})

    # Image operations

    def get_rgb(self, n, scl):
        rlow, rhigh = self.rscale
        glow, ghigh = self.gscale
        blow, bhigh = self.bscale

        section = ceil(n / self.ds.nz)
        z = n - (section - 1) * self.ds.nz - 1

        r = self.ds[scl][f"S{section:03d}"].sel(channel=self.channels[0], z=z).data
        r = (r.clip(rlow, rhigh) - rlow) / (rhigh - rlow)

        g = self.ds[scl][f"S{section:03d}"].sel(channel=self.channels[1], z=z).data
        g = (g.clip(glow, ghigh) - glow) / (ghigh - glow)

        b = self.ds[scl][f"S{section:03d}"].sel(channel=self.channels[2], z=z).data
        b = (b.clip(blow, bhigh) - blow) / (bhigh - blow)

        return (r, g, b)

    def get_image(self, data=None, x_range=None, y_range=None, x=None, y=None):

        rlow, rhigh = self.rscale
        glow, ghigh = self.gscale
        blow, bhigh = self.bscale

        if not data:
            data = {}

        scl = self.ds.compute_scale(x_range, y_range, res=self.resolution)
        self.scl = scl

        n = data.get("section", 1)
        self.current_section = n

        r, g, b = self.get_rgb(n, scl)

        try:
            chunksize = r.chunksize[0]
            y1, y2 = (
                y_range[0] // scl,
                y_range[1] // scl,
            )
            x1, x2 = (
                x_range[0] // scl,
                x_range[1] // scl,
            )
            x1, x2, y1, y2 = (
                int(x1) - chunksize // 2,
                int(x2) + chunksize // 2,
                int(y1) - chunksize // 2,
                int(y2) + chunksize // 2,
            )
            x1 = max(0, x1)
            x2 = min(x2, self.ds.shape[1] // scl)
            y1 = max(0, y1)
            y2 = min(y2, self.ds.shape[0] // scl)
        except:  # noqa: E722
            x1 = y1 = 0
            y2 = self.ds.shape[0] // scl
            x2 = self.ds.shape[1] // scl

        ir = r[y1:y2, x1:x2]
        ig = g[y1:y2, x1:x2]
        ib = b[y1:y2, x1:x2]
        ir, ig, ib = dask.compute([ir, ig, ib])[0]

        im = hv.RGB(
            (
                range(x1 * scl, x2 * scl, scl),
                range(y1 * scl, y2 * scl, scl),
                ir,
                ig,
                ib,
            )
        )
        return im

    def get_image_zoom(self, data=None, x=None, y=None):
        rlow, rhigh = self.rscale
        glow, ghigh = self.gscale
        blow, bhigh = self.bscale

        if not data:
            data = {}

        width = 160

        n = data.get("section", 1)
        scl = 1

        r, g, b = self.get_rgb(n, scl)

        try:
            x1 = int(x) - width // 2
            y1 = int(y) - width // 2
            x2 = int(x) + width // 2
            y2 = int(y) + width // 2
        except Exception:
            x1 = y1 = 0
            x2 = y2 = width

        if x1 < 0:
            x1 = 0
            x2 = x1 + width
        if y1 < 0:
            y1 = 0
            y2 = y1 + width
        if x2 >= self.ds.shape[1]:
            x2 = self.ds.shape[1] - 1
            x1 = x2 - width
        if y2 >= self.ds.shape[1]:
            y2 = self.ds.shape[1] - 1
            y1 = y2 - width

        ir = r[y1:y2, x1:x2]
        ig = g[y1:y2, x1:x2]
        ib = b[y1:y2, x1:x2]
        ir, ig, ib = dask.compute([ir, ig, ib])[0]

        im = hv.RGB(
            (
                range(0, width),
                range(0, width),
                ir,
                ig,
                ib,
            )
        )
        return im

    def update_slide(self, event):
        self.pipe.send({"section": event.obj.value})

    def mainview(self):
        image = hv.DynamicMap(self.get_image, streams=[self.pipe, self.range_xy])
        if self.buffer:
            res = regrid(image)
        else:
            res = hd.regrid(image)
        return res

    def miniview(self):
        image = hv.DynamicMap(self.get_image, streams=[self.pipe])
        res = hd.regrid(image)
        return res

    def zoomview(self):
        image = hv.DynamicMap(self.get_image_zoom, streams=[self.pipe, self.pointer])
        return image

    # Viewer

    def view(
        self,
        *,
        channels,
        rscale=None,
        gscale=None,
        bscale=None,
        percentile=98,
        show_miniview=True,
        height=600,
        resolution=900,
    ):
        self.channels = channels
        self.resolution = resolution
        self.rscale = rscale or self.ds.display_interval(channels[0], percentile)
        self.gscale = gscale or self.ds.display_interval(channels[1], percentile)
        self.bscale = bscale or self.ds.display_interval(channels[2], percentile)

        self.setup_streams()
        self.setup_controller(channels=channels)
        self.setup_template(height=height)

        tooltips = [
            ("x", "$x{(0)}"),
            ("y", "$y{(0)}"),
        ]
        hover = HoverTool(tooltips=tooltips)
        self.main = self.mainview().opts(
            clone=True,
            responsive=True,
            hooks=[remove_bokeh_logo],
            default_tools=[hover],
            title=f"Sample: {self.name}",
        )

        boxes = hv.Rectangles([])
        self.box_stream = streams.BoxEdit(
            source=boxes,
            styles={"fill_color": ["yellow", "red", "green", "blue", "cyan"]},
        )
        boxes = boxes.opts(hv.opts.Rectangles(active_tools=[], fill_alpha=0.5))

        overlay = hd.regrid(hv.Image([]), streams=[self.pointer])

        if show_miniview:
            mini = (
                self.miniview()
                .clone(link=False)
                .opts(
                    width=200,
                    height=200,
                    xaxis=None,
                    yaxis=None,
                    default_tools=[],
                    shared_axes=False,
                    hooks=[remove_bokeh_logo],
                )
            )
            zoom = self.zoomview().opts(
                width=200,
                height=200,
                xaxis=None,
                yaxis=None,
                default_tools=[],
                shared_axes=False,
                hooks=[remove_bokeh_logo],
            )
            RangeToolLink(mini, self.main, axes=["x", "y"])
            self.tmpl.add_panel(
                "A",
                pn.Row(
                    pn.panel(self.main * overlay * boxes),
                    pn.Column(pn.panel(mini), pn.panel(zoom)),
                    width=400,
                    height=280,
                    sizing_mode="scale_both",
                ),
            )
        else:
            self.tmpl.add_panel("A", pn.Row(pn.panel(self.main)))
        self.tmpl.add_panel("C", self.controller)
        return self.tmpl
