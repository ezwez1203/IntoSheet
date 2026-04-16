# Copyright 2025 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install mt3."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'mt3')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setuptools.setup(
    name='mt3',
    version=__version__,
    description='Multi-Task Multitrack Music Transcription',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/magenta/mt3',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={
        '': ['*.gin'],
    },
    scripts=[],
    python_requires='>=3.11,<3.12',
    install_requires=[
        'absl-py',
        'flax==0.7.5',
        'gin-config',
        'immutabledict',
        'jax==0.4.23',
        'jaxlib==0.4.23',
        'librosa',
        'mir_eval',
        'music21==9.1.0',
        'note-seq',
        'numpy>=1.22,<2.0',
        'pretty_midi',
        'scikit-learn',
        'scipy==1.11.4',
        'seqio==0.0.18',
        'setuptools==69.5.1',
        'optax==0.1.9',
        'chex==0.1.85',
        'orbax-checkpoint==0.4.4',
        't5',
        't5x',
        'tensorflow==2.14.1',
        'tensorflow-text==2.14.0',
        'tensorflow-hub==0.15.0',
        'tensorflow-datasets',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    keywords='music transcription machinelearning audio',
)
