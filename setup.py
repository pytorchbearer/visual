from setuptools import setup

version_dict = {}
exec(open("./visual/version.py").read(), version_dict)

import sys
if sys.version_info[0] >= 3:
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'A visualisation library for PyTorch using torchbearer'

setup(
    name='torchbearer_visual',
    version=version_dict['__version__'],
    packages=['visual', 'visual.models', 'tests', 'tests.models'],
    url='https://github.com/pytorchbearer/visual',
    download_url='https://github.com/pytorchbearer/visual/archive/' + version_dict['__version__'] + '.tar.gz',
    license='MIT',
    author='Ethan Harris',
    author_email='ewah1g13@ecs.soton.ac.uk',
    description='A visualisation library for PyTorch using torchbearer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['torch>=1.1.0', 'torchvision>=0.3.0', 'torchbearer'],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
)
