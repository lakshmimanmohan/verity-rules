# Step 7 validation results

Date: 2026-10-08

**Overall: NOT FULLY EVALUATED** (8 of 10 targets met, 0 failed, 2 not evaluable). Nothing was tuned. Degraded copies, anchor purity and the comparator baseline are report only.

| Target | Result | Value |
|---|---|---|
| customers: no composite_weight range above 0.05 | PASS | max range 0.000000 |
| products: no composite_weight range above 0.05 | PASS | max range 0.000000 |
| customers: likely_match purity >= 90% | PASS | 100.0% (30 of 30) |
| customers: likely_non_match purity >= 95% | PASS | 100.0% (30 of 30) |
| products: likely_match purity >= 90% | PASS | 100.0% (30 of 30) |
| products: likely_non_match purity >= 95% | PASS | 100.0% (30 of 30) |
| customers: learned hard AUC >= equal hard AUC | PASS | 1.000000 vs 1.000000 |
| customers: learned hard AUC >= uniqueness hard AUC - 0.005 | PASS | 1.000000 vs 1.000000 - 0.005 |
| products: learned hard AUC >= equal hard AUC | NOT EVALUABLE | nan vs nan (no hard negatives: all 331 candidate pairs are true duplicates) |
| products: learned hard AUC >= uniqueness hard AUC - 0.005 | NOT EVALUABLE | nan vs nan - 0.005 (no hard negatives: all 331 candidate pairs are true duplicates) |

## Stability across seeds

### Stability, customers, seeds 1 to 5

| Column | Min weight | Max weight | Range | Flag |
|---|---|---|---|---|
| first_name | 0.061711 | 0.061711 | 0.000000 |  |
| last_name | 0.081767 | 0.081767 | 0.000000 |  |
| email | 0.160069 | 0.160069 | 0.000000 |  |
| phone | 0.123084 | 0.123084 | 0.000000 |  |
| address | 0.152971 | 0.152971 | 0.000000 |  |
| city | 0.134767 | 0.134767 | 0.000000 |  |
| zip_code | 0.149048 | 0.149048 | 0.000000 |  |
| SSN | 0.136583 | 0.136583 | 0.000000 |  |

Column ranking the same across seeds: **yes**.
Ranking: email > address > zip_code > SSN > city > phone > last_name > first_name.

### Stability, products, seeds 1 to 5

| Column | Min weight | Max weight | Range | Flag |
|---|---|---|---|---|
| manufacturer_part_number | 0.424781 | 0.424781 | 0.000000 |  |
| member_part_number | 0.078360 | 0.078360 | 0.000000 |  |
| upc_ean | 0.427209 | 0.427209 | 0.000000 |  |
| manufacturer_code | 0.032603 | 0.032603 | 0.000000 |  |
| brand_code | 0.024452 | 0.024452 | 0.000000 |  |
| member_id | 0.012595 | 0.012595 | 0.000000 |  |

Column ranking the same across seeds: **yes**.
Ranking: upc_ean > manufacturer_part_number > member_part_number > manufacturer_code > brand_code > member_id.

## Calibration pairs against truth (seed 42)

### Calibration, customers

Separability: label **clear**, gap 0.627086, top-decile mean 0.988026, random-pair mean 0.36094.

| Category | Pairs | True duplicates | Share |
|---|---|---|---|
| likely_match (share true duplicates, target >= 90%) | 30 | 30 | 100.0% |
| likely_non_match (share true non-duplicates, target >= 95%) | 30 | 30 | 100.0% |
| unclear (share true duplicates, report only) | 40 | 32 | 80.0% |

### Calibration, products

Separability: label **clear**, gap 0.591477, top-decile mean 0.934922, random-pair mean 0.343445.

| Category | Pairs | True duplicates | Share |
|---|---|---|---|
| likely_match (share true duplicates, target >= 90%) | 30 | 30 | 100.0% |
| likely_non_match (share true non-duplicates, target >= 95%) | 30 | 30 | 100.0% |
| unclear (share true duplicates, report only) | 40 | 40 | 100.0% |

## Do the weights help?

Pair scoring is the package's weighted_pair_similarity with the weights swapped. learned = published composite_weight (v2). equal = 1 / number of content columns. v1 = composite_weight from learn_rules(reliability='v1'), seed 42. uniqueness = uniqueness_ratio normalised to sum 1, no reliability and no null-rate factor.

### Do the weights help, customers

Positives: 2678 true duplicate pairs. Random negatives: 5000. Hard negatives: 549 (candidate pairs: 3209, of which 549 not true duplicates).

| Weighting | AUC vs random negatives | AUC vs hard negatives |
|---|---|---|
| learned | 1.000000 | 1.000000 |
| equal | 1.000000 | 1.000000 |
| v1 | 1.000000 | 1.000000 |
| uniqueness | 1.000000 | 1.000000 |

Weights used:

| Column | learned | equal | v1 | uniqueness |
|---|---|---|---|---|
| first_name | 0.061711 | 0.125000 | 0.278815 | 0.056712 |
| last_name | 0.081767 | 0.125000 | 0.303300 | 0.075297 |
| email | 0.160069 | 0.125000 | 0.083984 | 0.174703 |
| phone | 0.123084 | 0.125000 | 0.072600 | 0.165144 |
| address | 0.152971 | 0.125000 | 0.067442 | 0.134505 |
| city | 0.134767 | 0.125000 | 0.059401 | 0.118469 |
| zip_code | 0.149048 | 0.125000 | 0.073753 | 0.147090 |
| SSN | 0.136583 | 0.125000 | 0.060705 | 0.128080 |

### Do the weights help, products

Positives: 401 true duplicate pairs. Random negatives: 5000. Hard negatives: 0 (candidate pairs: 331, of which 0 not true duplicates).

| Weighting | AUC vs random negatives | AUC vs hard negatives |
|---|---|---|
| learned | 1.000000 | nan |
| equal | 1.000000 | nan |
| v1 | 1.000000 | nan |
| uniqueness | 0.999636 | nan |

Weights used:

| Column | learned | equal | v1 | uniqueness |
|---|---|---|---|---|
| manufacturer_part_number | 0.424781 | 0.166667 | 0.464813 | 0.235031 |
| member_part_number | 0.078360 | 0.166667 | 0.209476 | 0.412869 |
| upc_ean | 0.427209 | 0.166667 | 0.138619 | 0.311886 |
| manufacturer_code | 0.032603 | 0.166667 | 0.090682 | 0.017873 |
| brand_code | 0.024452 | 0.166667 | 0.065201 | 0.013405 |
| member_id | 0.012595 | 0.166667 | 0.031210 | 0.008937 |

## Products anchor purity

### Anchor purity, products (relax off, so each level shows its own anchors)

Purity is the share of anchor pairs that are true duplicates. The last column is the level the default run used.

| Column | Anchors, min_agree 1 | Purity, 1 | Anchors, min_agree 2 | Purity, 2 | Level used |
|---|---|---|---|---|---|
| manufacturer_part_number | 210 | 100.0% | 21 | 100.0% | 1 |
| member_part_number | 329 | 100.0% | 173 | 100.0% | 2 |
| upc_ean | 235 | 100.0% | 23 | 100.0% | 1 |
| manufacturer_code | 331 | 100.0% | 191 | 100.0% | 2 |
| brand_code | 331 | 100.0% | 191 | 100.0% | 2 |
| member_id | 331 | 100.0% | 191 | 100.0% | 2 |

## Degraded copies of customers (report only, no cutoffs changed)

Truth file unchanged. Each copy was run through the CLI (seed 42). Blank fractions are of all rows; typo fractions are of the non-empty values left after blanking. Hard negatives are built from the degraded copy. nan means the set is empty. Random-negative AUC is added so a copy with no candidates still has a number.

| Copy | Separability gap | Label | likely_match true-duplicate share | unclear true-duplicate share | AUC learned (hard) | AUC equal (hard) | Hard negatives | AUC learned (random) | AUC equal (random) | Candidate pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.661548 | clear | 100.0% (30/30) | 92.5% (37/40) | 1.000000 | 1.000000 | 544 | 1.000000 | 1.000000 | 3198 |
| B | 0.664791 | clear | 100.0% (30/30) | 52.5% (21/40) | 1.000000 | 1.000000 | 2645 | 1.000000 | 1.000000 | 5281 |
| C | None | None | n/a (0/0) | n/a (0/0) | nan | nan | 0 | 0.890579 | 0.890883 | 0 |

- A: blank 25% of email, phone. Columns kept: first_name, last_name, email, phone, address, city, zip_code, SSN. Excluded: customer_id, source_system.
- B: blank 50% of email, phone; typos in 30% of first_name, last_name, address. Columns kept: first_name, last_name, email, phone, address, city, zip_code, SSN. Excluded: customer_id, source_system.
- C: blank 50% of every column except the record id. Columns kept: first_name, last_name, email, phone, address, city, zip_code, SSN. Excluded: customer_id, source_system.

## Comparator baseline

Mean field_similarity with each column's own comparison method, over pairs where both values are non-null. Non-duplicate set: the same 5000 random non-duplicate pairs used above. Duplicate set: all true duplicate pairs. Difference = duplicate mean minus non-duplicate mean, smallest first.

| Dataset | Column | Method | Non-dup mean | n | Dup mean | n | Difference |
|---|---|---|---|---|---|---|---|
| products | member_id | text_character | 0.749829 | 5000 | 0.746348 | 401 | -0.003480 |
| products | member_part_number | exact | 0.000000 | 5000 | 0.094763 | 401 | 0.094763 |
| customers | phone | text_character | 0.448758 | 3811 | 0.739765 | 2060 | 0.291007 |
| products | upc_ean | text_character | 0.518623 | 3873 | 0.852245 | 306 | 0.333623 |
| products | manufacturer_code | text_character | 0.661290 | 5000 | 1.000000 | 401 | 0.338710 |
| products | brand_code | text_character | 0.624922 | 5000 | 1.000000 | 401 | 0.375078 |
| customers | email | email | 0.447125 | 4617 | 0.831782 | 2444 | 0.384658 |
| customers | SSN | text_character | 0.496920 | 4498 | 0.981682 | 2391 | 0.484762 |
| customers | address | text_character | 0.370038 | 5000 | 0.989227 | 2678 | 0.619189 |
| customers | zip_code | text_character | 0.261416 | 5000 | 0.881617 | 2678 | 0.620202 |
| customers | city | text_character | 0.334152 | 5000 | 0.989853 | 2678 | 0.655700 |
| customers | first_name | text_character | 0.224205 | 4806 | 0.965573 | 2576 | 0.741368 |
| products | manufacturer_part_number | text_character | 0.235746 | 5000 | 0.983572 | 401 | 0.747826 |
| customers | last_name | text_character | 0.201423 | 4822 | 0.961701 | 2577 | 0.760278 |
