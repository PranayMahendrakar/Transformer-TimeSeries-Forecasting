from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith('#')]

setup(
    name='transformer-timeseries-forecasting',
    version='1.0.0',
    author='Pranay M Mahendrakar',
    author_email='pranaymahendrakar2001@gmail.com',
    description='Transformer-based time series forecasting: Informer, TFT, Autoformer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PranayMahendrakar/Transformer-TimeSeries-Forecasting',
    packages=find_packages(exclude=['tests*', 'notebooks*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={'dev': ['pytest', 'black', 'isort', 'flake8']},
)

