try:
    from setuptools import setup, find_namespace_packages
except ImportError:
    from distutils.core import setup, find_namespace_packages

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='explainitall',
    version='1.0.2',
    long_description=open('README.md', encoding='utf-8', errors='ignore').read(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(include=['explainitall', 'explainitall.*']),
    install_requires=REQUIREMENTS
)
