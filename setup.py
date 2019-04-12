from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'astropy',
    'matplotlib',
    'numpy',
    'scipy',
    'xmltodict',
    'holoviews',
    'zarr',
    'dask',
]

setup_requirements = ['pytest-runner', 'flake8']

test_requirements = ['coverage', 'pytest', 'pytest-cov', 'pytest-mock']

setup(
    author='IMAXT Team',
    maintainer='Eduardo Gonzalez Solares',
    maintainer_email='eglez@ast.cam.ac.uk',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='IMAXT Image Utilities',
    install_requires=requirements,
    license='GNU General Public License v3',
    long_description=readme,
    include_package_data=True,
    keywords='imaxt',
    name='imaxt-image',
    packages=find_packages(include=['imaxt*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.ast.cam.ac.uk/imaxt/imaxt-image',
    version='0.8.1',
    zip_safe=False,
    python_requires='>=3.5',
)
