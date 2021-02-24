from setuptools import setup
import os

github_url = f'https://github.com/{os.environ.get("GITHUB_REPOSITORY")}'

setup(
    setup_requires=['pbr'],
    pbr=True,
    tests_require=['nose'],
    test_suite='nose.collector',
    project_urls={
        'Source Code': github_url,
        'Bug Tracker': f'{github_url}/issues',
        'Documentation': 'https://soft-brownian-offset.readthedocs.io',
    },
)
