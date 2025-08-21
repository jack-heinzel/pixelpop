import setuptools

setuptools.setup(name='pixelpop',
        version='0.1',
        description='Meant for use with numpyro and jax, package for fitting the GW population with a nonparameteric binning scheme, where bins are correlated with only their nearest neighbors. Meant for inferring the GW population distribution nonparameterically in higher dimensions.',
        url='#',
        author='Jack Heinzel',
        install_requires=['numpy', 'scipy', 'jax', 'pandas', 'numpyro', 'gwpopulation', 'wcosmo', 'astropy', 'h5ify'],
        author_email='heinzelj@mit.edu',
        packages=[
            "pixelpop",
            "pixelpop.utils",
            "pixelpop.models",
            "pixelpop.result"
        ],
        package_dir={'pixelpop': 'pixelpop'},
        zip_safe=False)

# package_data={'gwpop_nearest_neighbor': ['utils/events/*.h5']},
