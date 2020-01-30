import setuptools

setuptools.setup(
    name = "optical_bloch",
    author = "Olivier Grasdijk",
    author_email = "olivier.grasdijk@yale.edu",
    description = "package for setting up and solving a ODE system of optical Bloch equations",
    url = "https://github.com/",
    packages = setuptools.find_packages(),
    install_requires = [
        'numpy',
        'sympy'
        ],
    python_requires = '>=3.6',
    version = "0.1"
)
