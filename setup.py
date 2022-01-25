from setuptools import setup,find_packages
import lixtools

setup(
    name='lixtools',
    description="""software tools for data collection/processing at LiX""",
    version=lixtools.__version__,
    author='Lin Yang',
    author_email='lyang@bnl.gov',
    license="BSD-3-Clause",
    url="https://github.com/NSLS-II-LIX/lixtools",
    packages=find_packages(),
    package_data={'': ['plate_label_template.html', 'template_report.ipynb']},
    include_package_data=True,
    install_requires=['py4xs', 'numpy', 'pandas', 'scipy>=1.6',
                      'python-barcode', 'matplotlib', 'pillow', 
                      'openpyxl>=3', 'qrcode'],
    python_requires='>=3.6',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='x-ray scattering',
)
