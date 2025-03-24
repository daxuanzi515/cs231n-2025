from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# fix bugs in latest version of cpython 
extensions = Extension(
    "im2col_cython",
    ["im2col_cython.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
)

setup(ext_modules=cythonize(extensions),)
print("Compile im2col_cython.pyx successfully!")
