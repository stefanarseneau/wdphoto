import setuptools

setuptools.setup(
    name="wdphoto",                     # This is the name of the package
    version="0.0.1",                        # The initial release version
    author="Stefan Arseneau",                     # Full name of the author
    description="Photometric Fitting for White Dwarfs",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["wdphoto"],             # Name of the python package
    package_dir={'':'wdphoto/src'},     # Directory of the source code of the package
    install_requires=[]                     # Install other dependencies if any
)