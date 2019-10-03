try:
    import holoviews as hv
except ImportError as e:
    msg = (
        'IMAXT-Image notebook requirements are not installed.\n\n'
        'Please pip install as follows:\n\n'
        '  pip install imaxt-image[notebook] --upgrade '
    )
    raise ImportError(str(e) + '\n\n' + msg)
