# Uniqueness and chance-corrected reliability: results

Report only, no tuning. Old = code at commit b2436de (before both changes). Seed 42.

## Change 1: uniqueness over non-null values

### customers

| Column | Uniqueness old | Uniqueness new | Weight old | Weight new | Change |
|---|---|---|---|---|---|
| first_name | 0.2670 | 0.2721 | 0.0617 | 0.0608 | -0.0009 |
| last_name | 0.3545 | 0.3607 | 0.0818 | 0.0804 | -0.0013 |
| email | 0.8225 | 0.8576 | 0.1601 | 0.1613 | +0.0013 |
| phone | 0.7775 | 0.8865 | 0.1231 | 0.1357 | +0.0126 |
| address | 0.6332 | 0.6332 | 0.1530 | 0.1479 | -0.0051 |
| city | 0.5577 | 0.5577 | 0.1348 | 0.1303 | -0.0045 |
| zip_code | 0.6925 | 0.6925 | 0.1490 | 0.1441 | -0.0050 |
| SSN | 0.6030 | 0.6377 | 0.1366 | 0.1396 | +0.0030 |

Candidate pairs: 3209 -> 3209. Separability gap: 0.627086 -> 0.624704.

### products

| Column | Uniqueness old | Uniqueness new | Weight old | Weight new | Change |
|---|---|---|---|---|---|
| manufacturer_part_number | 0.5260 | 0.5260 | 0.4248 | 0.4011 | -0.0237 |
| member_part_number | 0.9240 | 0.9240 | 0.0784 | 0.0740 | -0.0044 |
| upc_ean | 0.6980 | 0.7945 | 0.4272 | 0.4592 | +0.0320 |
| manufacturer_code | 0.0400 | 0.0400 | 0.0326 | 0.0308 | -0.0018 |
| brand_code | 0.0300 | 0.0300 | 0.0245 | 0.0231 | -0.0014 |
| member_id | 0.0200 | 0.0200 | 0.0126 | 0.0119 | -0.0007 |

Candidate pairs: 331 -> 331. Separability gap: 0.591477 -> 0.588165.

### Copy C (blank 50% of every column)

| Column | Uniqueness old | Uniqueness new | Weight old | Weight new | Change |
|---|---|---|---|---|---|
| first_name | 0.1732 | 0.3536 | 0.0624 | 0.0819 | +0.0195 |
| last_name | 0.2338 | 0.4753 | 0.0845 | 0.1111 | +0.0266 |
| email | 0.4405 | 0.9201 | 0.1551 | 0.0267 | -0.1284 |
| phone | 0.4125 | 0.9348 | 0.1338 | 0.0462 | -0.0876 |
| address | 0.3962 | 0.7920 | 0.1458 | 0.1930 | +0.0473 |
| city | 0.3673 | 0.7340 | 0.1351 | 0.1780 | +0.0429 |
| zip_code | 0.4130 | 0.8255 | 0.1519 | 0.1805 | +0.0286 |
| SSN | 0.3780 | 0.7986 | 0.1315 | 0.1826 | +0.0511 |

Candidate pairs: 0 -> 1675. Separability gap: None -> 0.771525.

Copy C candidate pairs: 0 before, 1675 after, so candidates now exist.

## Change 2: chance-corrected reliability (before = after change 1 only)

### customers

| Column | Reliability before | Reliability after | Weight before | Weight after | Change |
|---|---|---|---|---|---|
| first_name | 0.9655 | 0.9553 | 0.0608 | 0.0651 | +0.0043 |
| last_name | 0.9620 | 0.9523 | 0.0804 | 0.0862 | +0.0058 |
| email | 0.8314 | 0.6969 | 0.1613 | 0.1465 | -0.0149 |
| phone | 0.7395 | 0.5333 | 0.1357 | 0.1059 | -0.0297 |
| address | 0.9894 | 0.9830 | 0.1479 | 0.1591 | +0.0112 |
| city | 0.9896 | 0.9844 | 0.1303 | 0.1403 | +0.0101 |
| zip_code | 0.8815 | 0.8387 | 0.1441 | 0.1484 | +0.0044 |
| SSN | 0.9814 | 0.9632 | 0.1396 | 0.1484 | +0.0088 |

Separability gap: 0.624704 -> 0.629482.

### products

| Column | Reliability before | Reliability after | Weight before | Weight after | Change |
|---|---|---|---|---|---|
| manufacturer_part_number | 0.9908 | 0.9879 | 0.4011 | 0.4411 | +0.0400 |
| member_part_number | 0.1040 | 0.1040 | 0.0740 | 0.0816 | +0.0076 |
| upc_ean | 0.8572 | 0.7048 | 0.4592 | 0.4164 | -0.0428 |
| manufacturer_code | 1.0000 | 1.0000 | 0.0308 | 0.0340 | +0.0032 |
| brand_code | 1.0000 | 1.0000 | 0.0231 | 0.0255 | +0.0024 |
| member_id | 0.7726 | 0.0905 | 0.0119 | 0.0015 | -0.0104 |

Separability gap: 0.588165 -> 0.600243.

### Copy C

| Column | Reliability before | Reliability after | Weight before | Weight after | Change |
|---|---|---|---|---|---|
| first_name | 0.9607 | 0.9525 | 0.0819 | 0.0825 | +0.0006 |
| last_name | 0.9660 | 0.9584 | 0.1111 | 0.1120 | +0.0009 |
| email | 0.1231 | 0.1231 | 0.0267 | 0.0271 | +0.0004 |
| phone | 0.2277 | 0.2277 | 0.0462 | 0.0470 | +0.0007 |
| address | 0.9898 | 0.9837 | 0.1930 | 0.1950 | +0.0019 |
| city | 0.9846 | 0.9771 | 0.1780 | 0.1795 | +0.0015 |
| zip_code | 0.8880 | 0.8476 | 0.1805 | 0.1751 | -0.0054 |
| SSN | 0.9816 | 0.9624 | 0.1826 | 0.1819 | -0.0007 |

Separability gap: 0.771525 -> 0.771674.

## Copy H and the original customers: old code vs both changes

Copy H = customers with 50% of email and phone blanked, typos in 30% of first_name, last_name and address, 60% of zip_code and city scrambled with values from other rows, and 2 noise columns of 8-letter words and 2 of 10-digit strings (`tools/degrade.py`, seed 42). Truth file unchanged. Built with: `python tools/degrade.py data/customers.csv data/customers_degraded_H.csv --blank email,phone 0.5 --typo first_name,last_name,address 0.3 --scramble zip_code,city 0.6 --noise-cols 2`. The four noise columns are 100% unique, so Verity excludes them as primary keys and they get no weight.

### customers (original)

Truth: 2678 true duplicate pairs. Hard negatives (candidate pairs that are not true duplicates, built with the new uniqueness): 549 of 3209 candidate pairs. Kept columns: first_name, last_name, email, phone, address, city, zip_code, SSN. Excluded: customer_id, source_system. Separability: clear (gap 0.629482).

| Column | Uniqueness old | new | Null rate | Reliability old | new | Weight old | Weight new | Change |
|---|---|---|---|---|---|---|---|---|
| first_name | 0.2670 | 0.2721 | 0.0195 | 0.9655 | 0.9553 | 0.0617 | 0.0651 | +0.0034 |
| last_name | 0.3545 | 0.3607 | 0.0180 | 0.9620 | 0.9523 | 0.0818 | 0.0862 | +0.0045 |
| email | 0.8225 | 0.8576 | 0.0413 | 0.8314 | 0.6969 | 0.1601 | 0.1465 | -0.0136 |
| phone | 0.7775 | 0.8865 | 0.1232 | 0.7395 | 0.5333 | 0.1231 | 0.1059 | -0.0172 |
| address | 0.6332 | 0.6332 | 0.0000 | 0.9894 | 0.9830 | 0.1530 | 0.1591 | +0.0061 |
| city | 0.5577 | 0.5577 | 0.0000 | 0.9896 | 0.9844 | 0.1348 | 0.1403 | +0.0056 |
| zip_code | 0.6925 | 0.6925 | 0.0000 | 0.8815 | 0.8387 | 0.1490 | 0.1484 | -0.0006 |
| SSN | 0.6030 | 0.6377 | 0.0548 | 0.9814 | 0.9632 | 0.1366 | 0.1484 | +0.0118 |

AUC against hard negatives (same positives and negatives for every row):

| Weighting | AUC |
|---|---|
| learned (new) | 1.000000 |
| learned (old code) | 1.000000 |
| equal | 1.000000 |
| uniqueness only | 1.000000 |
| v1 | 1.000000 |

Weights used by each scheme:

| Column | learned (new) | learned (old code) | equal | uniqueness only | v1 |
|---|---|---|---|---|---|
| first_name | 0.0651 | 0.0617 | 0.1250 | 0.0555 | 0.2765 |
| last_name | 0.0862 | 0.0818 | 0.1250 | 0.0736 | 0.3000 |
| email | 0.1465 | 0.1601 | 0.1250 | 0.1751 | 0.0852 |
| phone | 0.1059 | 0.1231 | 0.1250 | 0.1810 | 0.0806 |
| address | 0.1591 | 0.1530 | 0.1250 | 0.1293 | 0.0656 |
| city | 0.1403 | 0.1348 | 0.1250 | 0.1139 | 0.0578 |
| zip_code | 0.1484 | 0.1490 | 0.1250 | 0.1414 | 0.0718 |
| SSN | 0.1484 | 0.1366 | 0.1250 | 0.1302 | 0.0625 |

### Copy H

Truth: 2678 true duplicate pairs. Hard negatives (candidate pairs that are not true duplicates, built with the new uniqueness): 5000 of 7301 candidate pairs (capped from 5027). Kept columns: first_name, last_name, email, phone, address, city, zip_code, SSN. Excluded: customer_id, noise_digits_1, noise_digits_2, noise_word_1, noise_word_2, source_system. Separability: clear (gap 0.619929).

| Column | Uniqueness old | new | Null rate | Reliability old | new | Weight old | Weight new | Change |
|---|---|---|---|---|---|---|---|---|
| first_name | 0.4785 | 0.4878 | 0.0195 | 0.8824 | 0.8473 | 0.1414 | 0.1626 | +0.0212 |
| last_name | 0.5617 | 0.5718 | 0.0180 | 0.8880 | 0.8602 | 0.1674 | 0.1938 | +0.0265 |
| email | 0.4422 | 0.9252 | 0.5222 | 0.8299 | 0.1536 | 0.0599 | 0.0272 | -0.0327 |
| phone | 0.4158 | 0.9486 | 0.5620 | 0.7286 | 0.1593 | 0.0453 | 0.0266 | -0.0188 |
| address | 0.8095 | 0.8095 | 0.0000 | 0.9626 | 0.9406 | 0.2662 | 0.3056 | +0.0394 |
| city | 0.4422 | 0.4422 | 0.0000 | 0.4358 | 0.1467 | 0.0658 | 0.0260 | -0.0398 |
| zip_code | 0.5202 | 0.5202 | 0.0000 | 0.3552 | 0.1234 | 0.0631 | 0.0258 | -0.0374 |
| SSN | 0.6030 | 0.6377 | 0.0548 | 0.9798 | 0.9608 | 0.1908 | 0.2324 | +0.0416 |

AUC against hard negatives (same positives and negatives for every row):

| Weighting | AUC |
|---|---|
| learned (new) | 0.999998 |
| learned (old code) | 0.999994 |
| equal | 0.996265 |
| uniqueness only | 0.993465 |
| v1 | 0.995483 |

Weights used by each scheme:

| Column | learned (new) | learned (old code) | equal | uniqueness only | v1 |
|---|---|---|---|---|---|
| first_name | 0.1626 | 0.1414 | 0.1250 | 0.0913 | 0.4143 |
| last_name | 0.1938 | 0.1674 | 0.1250 | 0.1070 | 0.3057 |
| email | 0.0272 | 0.0599 | 0.1250 | 0.1732 | 0.0383 |
| phone | 0.0266 | 0.0453 | 0.1250 | 0.1775 | 0.0360 |
| address | 0.3056 | 0.2662 | 0.1250 | 0.1515 | 0.0701 |
| city | 0.0260 | 0.0658 | 0.1250 | 0.0828 | 0.0383 |
| zip_code | 0.0258 | 0.0631 | 0.1250 | 0.0974 | 0.0451 |
| SSN | 0.2324 | 0.1908 | 0.1250 | 0.1193 | 0.0522 |

