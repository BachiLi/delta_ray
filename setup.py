from distutils.core import setup, Extension
import pybind11
import os
import platform
import os

if 'CC' not in os.environ:
    os.environ['CC'] = 'clang'
if 'CXX' not in os.environ:
    os.environ['CXX'] = 'clang++'

cc = os.environ['CC']
if cc[:len('ccache')] != 'ccache':
    os.environ['CC'] = 'ccache ' + cc

extra_compile_args = ['-std=c++1z',
                      '-Wno-missing-braces',
                      '-fvisibility=hidden',
                      '-ffast-math']
extra_link_args = ['-lz', '-ffast-math']
if platform.system() == 'Darwin':
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')

module = Extension('delta_ray',
                   include_dirs = [pybind11.get_include(), '/usr/local/include/embree3'],
                   extra_compile_args = extra_compile_args,
                   extra_link_args = extra_link_args,
                   sources = ['aabb.cpp', 
                              'delta_ray.cpp',
                              'edge_tree.cpp',
                              'parallel.cpp',
                              'scene.cpp'],
                   depends = ['aabb.h',
                              'autodiff.h',
                              'camera.h',
                              'distribution.h',
                              'edge.h',
                              'edge_tree.h',
                              'intersect.h',
                              'light.h',
                              'material.h',
                              'parallel.h',
                              'sample.h',
                              'scene.h',
                              'shape.h',
                              'transform.h',
                              'vector.h',
                              'progress_reporter.h',
                              'pathtrace.h',
                              'ltc.inc',
                              'line_clip.h'],
                   library_dirs = ['/usr/local/lib'],
                   libraries = ['embree3'])

setup(name = 'delta_ray',
      version = '1.0',
      description = 'Delta Ray',
      ext_modules = [module])

module = Extension('load_serialized',
                   include_dirs = [pybind11.get_include()],
                   extra_compile_args = extra_compile_args,
                   extra_link_args = extra_link_args,
                   sources = ['load_serialized.cpp'],
                   depends = [])

setup(name = 'load_serialized',
      version = '1.0',
      description = 'Load serialized',
      ext_modules = [module])
