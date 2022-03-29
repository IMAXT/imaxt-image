import json
import s3fs

from pathlib import Path


def get_mc_config(alias):
    p = Path("~/.mc/config.json").expanduser()
    with open(p, "r") as fh:
        config = json.loads(fh.read())
    return config["aliases"][alias]


def list_s3_store(alias="imaxtgw", instrument="stpt"):
    config = get_mc_config(alias)
    s3 = s3fs.S3FileSystem(
        key=config["accessKey"],
        secret=config["secretKey"],
        client_kwargs={"endpoint_url": config["url"]},
    )
    s3list = []
    for ppath in [
        "processed",
        "processed0",
        "processed1",
        "processed2",
        "processed3",
        "processed4",
    ]:
        try:
            this = f"{ppath}/{instrument}"
            s3list = s3list + [a.replace(this+'/', '') for a in s3.ls(this)]
        except PermissionError:
            pass
    return s3list


def get_s3_store(name, alias="imaxtgw", instrument="stpt", extra="mos.zarr"):
    config = get_mc_config(alias)
    s3 = s3fs.S3FileSystem(
        key=config["accessKey"],
        secret=config["secretKey"],
        client_kwargs={"endpoint_url": config["url"]},
    )
    s3store = None
    for ppath in [
        "processed",
        "processed0",
        "processed1",
        "processed2",
        "processed3",
        "processed4",
    ]:
        s3path = f"{ppath}/{instrument}/{name}/{extra}"
        if s3.exists(s3path):
            s3store = s3.get_mapper(s3path)
            break
    return s3store


