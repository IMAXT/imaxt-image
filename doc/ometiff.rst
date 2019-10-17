Reading OME-TIFF files
----------------------

The module :ref:`imaxt-image.image.TiffImage <tiffimage>` is used to read OME-TIFF 
files. The following example reads an image data and metadata:

.. code-block:: python

   from pathlib import Path
   from imaxt_image.io import TiffImage

   filename = Path('/some/directory')
   img = TiffImage(filename / 'image.tif')

   # Dimensions of image
   shape = img.shape

   # Image OME-TIFF metadata
   ome = img.metadata

   # Image data array
   arr = img.asarray()

