try:
    from setuptools import setup, find_namespace_packages
except ImportError:
    from distutils.core import setup, find_namespace_packages

with open("requirements.txt", encoding="utf-8") as f:
    REQUIREMENTS = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name='explainitall',
    version='1.0.2',
    long_description=open('README.md', encoding='utf-8', errors='ignore').read(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(include=['explainitall', 'explainitall.*']),
    # inseq 0.5.0 не поддерживает Python 3.12+
    python_requires='>=3.9,<3.12',
    install_requires=REQUIREMENTS,
    extras_require={
        # fast_tuning.trainers.DenceKerasTrainer написан под API Keras 2
        'keras': ['tensorflow>=2.12,<2.16'],
        'dev': ['pytest'],
    },
)
