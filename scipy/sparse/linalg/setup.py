#!/usr/bin/env python
from __future__ import division, print_function, absolute_import


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('linalg',parent_package,top_path)

    config.add_subpackage(('isolve'))
    config.add_subpackage(('dsolve'))
    config.add_subpackage(('eigen'))

    config.add_data_dir('tests')
    config.add_data_dir('benchmarks')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
