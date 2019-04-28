import os
from distutils.core import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(
    name='coptim',
    version='0.1',
    author='Cody Mazza-Anthony',
    author_email='cmazzaanthony@gmail.com',
    packages=['coptim', ],
    license='MIT License',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
    ],
    long_description=read('README.rst'),
)
