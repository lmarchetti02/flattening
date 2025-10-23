from pathlib import Path
from setuptools import setup, Extension

# Path to the C source file
c_file = Path("src/flattening/c_library/interpolate.c")

# Define an Extension
ext_modules = [
    Extension(
        name="flattening.c_library.lib_flattening",  # This becomes your_library/myclib.so
        sources=[str(c_file)],
        extra_compile_args=["-O3", "-fPIC", "-fopenmp", "-shared", "-lm"],
    )
]

setup(
    package_dir={"": "src"},
    packages=["flattening"],
    ext_modules=ext_modules,
)
