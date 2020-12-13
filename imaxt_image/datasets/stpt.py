import xarray as xr
import xtiff
import zarr
from pathlib import Path


class STPTSection:
    def __init__(self, ds, meta):
        self.ds = ds.astype("uint16")
        self.meta = meta

    def to_tiff(self, name, dir=None):
        section = self.meta["section"]
        group = self.meta["group"]
        if dir is None:
            dir = Path(".")
        else:
            dir = Path(dir)
        out = f"{dir}/{name}_{section}_{group or 'l.1'}.ome.tiff"
        xtiff.to_tiff(self.ds.data, out)
        print("Written", out)

    def __getitem__(self, item):
        return STPTSection(self.ds[:, :, item[0], item[1]], self.meta)


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

    def _read_dataset(self, clip=(0, 2 ** 16 - 1), multiplier=1000):
        if self.scale == 1:
            self.group = ""
        else:
            self.group = f"l.{self.scale}"
        self.ds = xr.open_zarr(f"{self.mos}", group=self.group)
        self.ds = self.ds.sel(type="mosaic") * self.bscale + self.bzero
        self.ds = self.ds.clip(clip[0], clip[1]) * multiplier

    def sel(self, scale=1, **kwargs):
        return STPTDataset(self.name, self.path, scale)

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
        }
        return STPTSection(self.ds[item], meta)
