from setuptools import setup, find_packages

config = {
    'description': 'Automated Modelling in Biological Evidence-based Research',
    'download_url': 'https://github.com/zj-zhang/AMBER',
    'version': '0.1.3',
    'packages': find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    'include_package_data': True,
    'package_data': {'amber': ['./amber/resources/gui/*']},
    'setup_requires': [],
    'install_requires': [
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
        'tensorflow >=1.9.0',
        'seaborn >=0.9.0',
        ],
    'dependency_links': [],
    'scripts': ['bin/Amber'],
    'name': 'amber-automl',
}

if __name__== '__main__':
    setup(**config)
