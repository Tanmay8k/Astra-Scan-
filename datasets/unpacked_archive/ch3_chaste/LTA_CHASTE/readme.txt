
Chandrayaan-3 Lander ChaSTE (CHT) PDS4 Data Archive

Contents:
1. Introduction
2. Archive Contents
3. Contacts for more information


1.  INTRODUCTION

This is readme file containing details about the PDS4 Data Archive of Lander ChaSTE instrument
of Chandrayaan-3 Lander


2.  ARCHIVE CONTENTS

Under cht instrument id wise data collection, data is organized under each directory
defined by PDS4. Data is organized based on the levels of data processing and data
products definitions. data products are organized into year, month and day wise as 
yyyymmdd. The year month and day wise directory contains PDS4 data products including 
PDS4 Label products.

Below is detailed ChaSTE data archive structure

cht
|--bundle_cht.xml
|--calibration
|   |--ch3_cht_res_gain_offset.csv
|   |--ch3_cht_res_gain_offset.xml
|   |--ch3_cht_res_temp_cal_coe.csv
|   |--ch3_cht_res_temp_cal_coe.xml
|   |--collection_calibration_inventory.csv
|   |--collection_calibration_inventory.xml
|
|--data
|  |
|  |--raw
|  |  |--yyyymmdd
|  |     |-- data_product
|  |--calibrated
|	 |--yyyymmdd
|        |-- data_product
|
|--collection_data_raw_inventory.csv
|--collection_data_raw_inventory.xml
|--collection_data_calibrated_inventory.csv
|--collection_data_calibrated_inventory.csv
|--document
|  |--ch3_cht_pds_dp_archive_sis.pdf
|  |--ch3_cht_pds_dp_archive_sis.xml
|  |--ch3_cht_cal.pdf
|  |--ch3_cht_cal.xml
|  |--collection_document_inventory_cht.csv
|  |--collection_document_inventory_cht.xml
|
|--miscallaneous_data |--Operation phase
|                     |--Re-insertion phase
|--readme.txt

The details about the data_product file naming conventions and formats 
please refer section 6.5 and 7 in data products archive software interface 
specification document (ch3_cht_pds_dp_archive_sis.pdf) under document 
collection.


2.1.  The observational data collections

Under the data directory there is one separate subdirectory based on the 
defined PDS4 data products levels calibrated. Under each 
sub directories, there are again sub directory based on year, month and day 
wise. The year, month and day wise subdirectory contains PDS4 data products 
mainly image for two sensors defined as left and right and associated 
label files in XML format.

2.2.  The Document Collection

The document directory contains documentation to help the user understand and use 
the archive data. The document collection contains data products and archive software 
interface specification document data product contains end to end to description about 
the mission, payload, data products (content, format, file naming convention) and 
archive structure as per PDS4 standard. 

3.  CONTACTS FOR MORE INFORMATION
=============================================================================
1. Amitabh
   Deputy Project Director (DPD) Data Products
   Space Applications Centre
   Ahmedabad
   Phone: 07926914727
   Email id: amitabh@sac.isro.gov.in

3. Ajay Kumar Prashar
   PDS4 Technical Support
   Space Applications Centre
   Ahmedabad
   Phone: 07926914764
   Email id: ajay_prashar@sac.isro.gov.in

4. Sreenath
   Operations Support Team
   ISTRAC
   Bengaluru
   Phone: 08028094416, 08022029173
   Email id: sreenath@istrac.gov.in
================================================================================
