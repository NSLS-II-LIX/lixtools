from setuptools import setup
import lixtools

setup(
    name='lixtools',
    description="""software tools for data collection/processing at LiX""",
    version=lixtools.__version__,
    author='Lin Yang',
    author_email='lyang@bnl.gov',
    license="BSD",
    url="",
    packages=['lixtools'],
    install_requires=['py4xs', 'numpy', 'pandas', 
                      'python-barcode', 'matplotlib', 'pillow', 'blabel',
                      'openpyxl>=3', 'xlrd'],
    python_requires='>=3.6',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='x-ray scattering',
)
