from pathlib import Path

import numpy as np
import xarray as xr
import xtiff
import zarr
from scipy import ndimage as ndi


def image_shift(data, offsets, block_info=None):
    if block_info is not None:
        ndim = len(data.shape)
        dd = data.squeeze()
        try:
            ndd = ndi.shift(dd, offsets)
        except Exception:
            ndd = data
        for _i in range(ndim - 2):
            ndd = ndd[np.newaxis, :]
    else:
        ndd = data
    return ndd


class STPTSection:
    def __init__(self, ds, meta):
        self.ds = ds.astype("uint16")
        self.ds.attrs = meta["attrs"]
        self.offsets = meta["offsets"]
        if self.offsets is not None:
            self.offsets = np.nan_to_num(meta["offsets"])
        self.meta = meta

    def to_tiff(self, name, dir=None):
        section = self.meta["section"]
        group = self.meta["group"]
        if dir is None:
            dir = Path(".")
        else:
            dir = Path(dir)
        out = f"{dir}/{name}_{section}_{group or 'l.1'}.ome.tiff"
        xtiff.to_tiff(self.data, out)
        print("Written", out)

    def sel(self, **kwargs):
        return STPTSection(self.ds.sel(**kwargs), self.meta)

    @property
    def data(self):
        dd = self.ds.data
        if self.offsets is not None:
            ndim = len(self.ds.data.shape)
            depth = np.abs(self.offsets).max() + 100
            depth = [0] * (ndim - 2) + [depth] * 2
            ndd = dd.map_overlap(image_shift, depth=depth, offsets=self.offsets)
        else:
            ndd = dd
        return ndd

    @property
    def values(self):
        return self.ds.values

    @property
    def attrs(self):
        return self.ds.attrs

    def __getitem__(self, item):
        yslice, xslice = item
        scale = self.meta["scale"]
        ystart, yend = yslice.start // scale, yslice.stop // scale
        xstart, xend = xslice.start // scale, xslice.stop // scale
        return STPTSection(self.ds[:, :, ystart:yend, xstart:xend], self.meta)

    def __repr__(self):
        return self.ds.__repr__()

    def _repr_html_(self):
        return self.ds._repr_html_()


class STPTDataset:
    def __init__(self, name, path, scale=1):
        self.path = Path(path)
        self.name = Path(name)
        self.scale = scale
        self.mos = self.path / self.name / "mos.zarr"

        self._read_bscale_bzero()
        self._read_dataset()

    def _read_bscale_bzero(self):
        z = zarr.open(f"{self.mos}", mode="r")
        self.bscale = z.attrs["bscale"]
        self.bzero = z.attrs["bzero"]

    def _read_metadata(self):
        ds0 = xr.open_zarr(f"{self.mos}")
        return ds0.attrs

    def _read_offsets(self):
        offsets = {
            s: [dx, dy]
            for s, dx, dy in zip(
                list(self.ds),
                self.attrs["cube_reg"]["abs_dx"],
                self.attrs["cube_reg"]["abs_dy"],
            )
        }
        return offsets

    def _read_dataset(self, clip=(0, 2 ** 16 - 1), multiplier=1000):
        if self.scale == 1:
            self.group = ""
        else:
            self.group = f"l.{self.scale}"
        self.ds = xr.open_zarr(f"{self.mos}", group=self.group)
        self.ds = self.ds.sel(type="mosaic") * self.bscale + self.bzero
        self.ds = self.ds.clip(clip[0], clip[1]) * multiplier
        if self.scale > 1:
            self.ds.attrs = self._read_metadata()
        try:
            self.offsets = self._read_offsets()
            self.offsets = [item / self.scale for item in self.offsets]
        except Exception:
            self.offsets = None

    def sel(self, scale=1, **kwargs):
        return STPTDataset(self.name, self.path, scale)

    @property
    def attrs(self):
        return self.ds.attrs

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, key):
        if isinstance(key, int):
            item = list(self.ds)[key]
        else:
            item = key
        meta = {
            "name": self.name,
            "scale": self.scale,
            "group": self.group,
            "section": item,
            "attrs": self.attrs,
            "offsets": self.offsets,
        }
        return STPTSection(self.ds[item], meta)

    def __repr__(self):
        return self.ds.__repr__()

    def _repr_html_(self):
        return self.ds._repr_html_()
