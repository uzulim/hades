from setuptools import setup

## Get version information from _version.py
import re

VERSIONFILE = "_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Use README.md as the package long description
with open("README.md") as f:
    long_description = f.read()


# get requirements
def requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip()]


setup(
    name="hades",
    version=verstr,
    description="HADES: Fast Singularity Detection with Kernel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Uzu Lim, Harald Oberhauser, Vidit Nanda",
    author_email="finnlimsh@gmail.com",
    license="BSD-3-Clause",
    packages=[],
    install_requires=requirements(),
    extras_require={
        "testing": [],
        "examples": [],
    },
    python_requires=">=3.8, <=3.12",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="topological data analysis, geometric data analysis, anomaly detection",
)