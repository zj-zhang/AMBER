from setuptools import setup, find_packages

config = {
    'description': 'Automated Modelling in Biological Evidence-based Research',
    'download_url': 'https://github.com/zj-zhang/AMBER',
    'version': '0.1.4',
    'packages': find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    'include_package_data': True,
    'setup_requires': [],
    'install_requires': [
        'numpy',
        #'scipy',
        #'scikit-learn',
        'matplotlib',
        'tqdm',
        'packaging',
        #'seaborn',
        ],
    'dependency_links': [],
    'name': 'amber-automl',
}

if __name__== '__main__':
    setup(**config)
