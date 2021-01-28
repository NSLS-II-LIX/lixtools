from setuptools import setup,find_packages
import lixtools

setup(
    name='lixtools',
    description="""software tools for data collection/processing at LiX""",
    version=lixtools.__version__,
    author='Lin Yang',
    author_email='lyang@bnl.gov',
    license="BSD",
    url="",
    packages=find_packages(),
    package_data={'': ['plate_label_template.html', 'template_report.ipynb']},
    include_package_data=True,
    install_requires=['py4xs', 'numpy', 'pandas', 
                      'python-barcode', 'matplotlib', 'pillow', 
                      'openpyxl>=3', 'xlrd', "qrcode"],
    python_requires='>=3.6',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='x-ray scattering',
)
