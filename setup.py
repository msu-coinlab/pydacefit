from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


__name__ = "pydacefit"
__author__ = "Julian Blank"
__version__ = '1.0.0'
__url__ = "https://github.com/msu-coinlab/pydacefit"

setup(
    name=__name__,
    version=__version__,
    author=__author__,
    author_email="blankjul@egr.msu.edu",
    description="Surrogate Toolbox for Python",
    long_description=readme(),
    url=__url__,
    license='Apache License 2.0',
    keywords="metamodel, surrogate, response surface",
    install_requires=['numpy'],
    packages=find_packages(exclude=['tests', 'docs']),
    include_package_data=True,
    platforms='any',
)
