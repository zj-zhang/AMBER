from setuptools import setup, find_packages

config = {
    'description': 'Automated Modelling in Biological Evidence-based Research',
    'download_url': 'https://github.com/zj-zhang/DeepAmbre',
    'version': '0.1.0',
    'packages': find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    'include_package_data': True,
    'package_data': {'deepambre': ['./amber/resources/gui/*']},
    'setup_requires': [],
    'install_requires': [
        'numpy', 
        'matplotlib', 
        'scipy', 
        'tensorflow>=1.9.0,<2.0.0', 
        'keras>=2.0.0', 
        'seaborn>=0.9.0',
        'networkx'
        ],
    'dependency_links': [],
    'scripts': ['bin/Amber'],
    'name': 'Amber',
}

if __name__== '__main__':
    setup(**config)
