try:
    from setuptools import setup, find_namespace_packages
except ImportError:
    from distutils.core import setup, find_namespace_packages

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='explainitall',
    version='1.0.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(include=['explainitall', 'explainitall.*']),
    install_requires=REQUIREMENTS
)