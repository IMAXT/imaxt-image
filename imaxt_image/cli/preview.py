from argparse import ArgumentParser, FileType

from matplotlib import pyplot as plt

from imaxt_image.image import TiffImage
from imaxt_image.image.scaling import zscale


def main():
    parser = ArgumentParser()
    parser.add_argument('filename', type=FileType('r'))
    args = parser.parse_args()

    img = TiffImage(args.filename.name).asarray()
    vmin, vmax = zscale(img)
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.show()
