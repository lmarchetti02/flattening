from setuptools import setup, Extension


module = Extension(
    "flattening.interpolate",
    sources=["src/flattening/interpolate.c"],
    extra_compile_args=["-O3", "-fPIC", "-fopenmp", "-shared"],
    extra_link_args=["-fopenmp"],
    libraries=["m"],
)

setup(
    name="flattening",
    version="0.1",
    ext_modules=[module],
)
