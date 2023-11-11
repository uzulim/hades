from setuptools import setup

def requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip()]

setup(
    name="hades",
    version="0.1.0",
    packages=["hades"],
    python_requires='>=3.8, <=3.11',
    install_requires=requirements(),
    package_data={'hades': ['mmd_data.json', 'mmd_data_big.json']},

    author='Uzu Lim',
    author_email='finnlimsh@gmail.com',
    description='Fast singularity detection',
    license='BSD-3-Clause',
    url='https://github.com/uzulim/hades',
)