from setuptools import find_packages, setup

requirements = [
    "astropy",
    "numpy",
    "scipy",
    "xmltodict",
    "imagecodecs",
    "zarr",
    "dask",
    "xarray",
    "scikit-image>=0.18",
    "xtiff",
]

extras_requirements = {"pyviz": ["matplotlib", "holoviews", "datashader"]}

setup_requirements = ["pytest-runner", "flake8"]

test_requirements = ["coverage", "pytest", "pytest-cov", "pytest-mock"]

setup(
    author="IMAXT Team",
    maintainer="Eduardo Gonzalez Solares",
    maintainer_email="eglez@ast.cam.ac.uk",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.7",
    ],
    description="IMAXT Image Utilities",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description="IMAXT Image Utils",
    include_package_data=True,
    keywords="imaxt",
    name="imaxt-image",
    packages=find_packages(include=["imaxt*"]),
    setup_requires=setup_requirements,
    extras_requires=extras_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.ast.cam.ac.uk/imaxt/imaxt-image",
    version="0.16.2",
    zip_safe=False,
    python_requires=">=3.7",
)
