from setuptools import setup, find_packages

setup(
    name='McDevol',
    version='0.1.0',
    packages=find_packages(),
    data_files=[('example', ['checkm2/testrun/abundance.tsv', 'checkm2/testrun/contigs.fasta'])],
    include_package_data=True,
    url='https://github.com/soeding/mcdevol',
    license='',
    install_requires=(),
    author='Yazhini Arangasamy',
    scripts=['bin/mcdevol'],
    author_email='yazhini@mpinat.mpg.de',
    description='Mcdevol- Metagenomic contigs binning'
)
