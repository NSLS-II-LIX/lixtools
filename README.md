# lixtools
Software tools for data collection and processing at the LiX beamline

Change log:

2020Oct:
Initial release

2020Nov: 
1. add plate name to holder name during Opentron transfer;
2. initial implementation of transfer using both pipetters

2020Dec:
moved some components from py4xs;
pdf report on static solution scattering data

2021Jan:
bug fixes for gen_pdf_report();
new module for producing structural models; support for DENSS

2020Feb:
mailin: take existing labware dictionary for generate_docs()

2020Mar:
other atsas tools for model_data(); chi2 cutoff 

2021Jun:
h5sol_ref for absolute intensity scaling

2021Aug:
revised h5sol_ref to deal with monitor counts in a different stream in h5;
more checks in spreadsheet validation;
moved webcam.py and ot2.py under lixtools/inst to avoid dependency check

2021Oct:
option for trans_mode.external for HPLC data;
first version of h5xs_scan

2022Jan:
update after implementing functions for h5xs_scan

2022Apr:
reorganized notebooks

