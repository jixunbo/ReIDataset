# ReID Dataset
In this repo 5 datasets for Person Re-Identification are uploaded.

Market-1501, MOT17Det and MOT17Labels are downloaded orignal datasets.

MOT17ReID-query2 and MOT17ReID-query5 are the person's bounding boxes
extracted and croped from MOT17 Dataset and being labeled. The dataset has 
a similar structure as Market-1501. 

In total we have 501 persons, in which 402 persons are allocated to 
train-val split, and 99 persons are allocated to test split(query-gallery).

query2 and query5 means each person has 2 or 5 query images.

prepareMOT.py is the code to extract MOT17ReID-query2 from original MOT17Det and MOT17Labels

