# Step 7 validation results

Date: 2026-10-08

**Overall: NOT FULLY EVALUATED** (8 of 10 targets met, 0 failed, 2 not evaluable). Nothing was tuned. Degraded copies, anchor purity and the comparator baseline are report only.

| Target | Result | Value |
|---|---|---|
| customers: no composite_weight range above 0.05 | PASS | max range 0.000700 |
| products: no composite_weight range above 0.05 | PASS | max range 0.002547 |
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
| first_name | 0.055404 | 0.055497 | 0.000093 |  |
| last_name | 0.073012 | 0.073124 | 0.000112 |  |
| email | 0.137470 | 0.138170 | 0.000700 |  |
| phone | 0.168031 | 0.168207 | 0.000176 |  |
| address | 0.145088 | 0.145161 | 0.000073 |  |
| city | 0.126950 | 0.127038 | 0.000088 |  |
| zip_code | 0.157803 | 0.157886 | 0.000083 |  |
| SSN | 0.135488 | 0.135618 | 0.000130 |  |

Column ranking the same across seeds: **yes**.
Ranking: phone > zip_code > address > email > SSN > city > last_name > first_name.

### Stability, products, seeds 1 to 5

| Column | Min weight | Max weight | Range | Flag |
|---|---|---|---|---|
| manufacturer_part_number | 0.309190 | 0.310272 | 0.001082 |  |
| member_part_number | 0.247535 | 0.250082 | 0.002547 |  |
| upc_ean | 0.396875 | 0.398306 | 0.001431 |  |
| manufacturer_code | 0.024266 | 0.024352 | 0.000086 |  |
| brand_code | 0.018200 | 0.018264 | 0.000064 |  |
| member_id | 0.001159 | 0.001388 | 0.000229 |  |

Column ranking the same across seeds: **yes**.
Ranking: upc_ean > manufacturer_part_number > member_part_number > manufacturer_code > brand_code > member_id.

## Calibration pairs against truth (seed 42)

### Calibration, customers

Separability (debug only, no label): gap 0.896937, top-decile mean 0.991553, random-pair mean 0.094616.

| Category | Pairs | True duplicates | Share |
|---|---|---|---|
| likely_match (share true duplicates, target >= 90%) | 30 | 30 | 100.0% |
| likely_non_match (share true non-duplicates, target >= 95%) | 30 | 30 | 100.0% |
| unclear (share true duplicates, report only) | 40 | 15 | 37.5% |

### Calibration, products

Separability (debug only, no label): gap 0.78934, top-decile mean 0.979024, random-pair mean 0.189684.

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
| first_name | 0.055405 | 0.125000 | 0.276484 | 0.055543 |
| last_name | 0.073037 | 0.125000 | 0.300032 | 0.073650 |
| email | 0.138131 | 0.125000 | 0.085225 | 0.175094 |
| phone | 0.168070 | 0.125000 | 0.080561 | 0.180991 |
| address | 0.145082 | 0.125000 | 0.065636 | 0.129285 |
| city | 0.126965 | 0.125000 | 0.057810 | 0.113871 |
| zip_code | 0.157795 | 0.125000 | 0.071777 | 0.141381 |
| SSN | 0.135515 | 0.125000 | 0.062474 | 0.130186 |

### Do the weights help, products

Positives: 401 true duplicate pairs. Random negatives: 5000. Hard negatives: 0 (candidate pairs: 331, of which 0 not true duplicates).

| Weighting | AUC vs random negatives | AUC vs hard negatives |
|---|---|---|
| learned | 1.000000 | nan |
| equal | 1.000000 | nan |
| v1 | 1.000000 | nan |
| uniqueness | 1.000000 | nan |

Weights used:

| Column | learned | equal | v1 | uniqueness |
|---|---|---|---|---|
| manufacturer_part_number | 0.310227 | 0.166667 | 0.456070 | 0.225314 |
| member_part_number | 0.247826 | 0.166667 | 0.205536 | 0.395799 |
| upc_ean | 0.398108 | 0.166667 | 0.154819 | 0.340336 |
| manufacturer_code | 0.024346 | 0.166667 | 0.088977 | 0.017134 |
| brand_code | 0.018259 | 0.166667 | 0.063975 | 0.012851 |
| member_id | 0.001232 | 0.166667 | 0.030623 | 0.008567 |

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

| Copy | Separability gap | likely_match true-duplicate share | unclear true-duplicate share | AUC learned (hard) | AUC equal (hard) | Hard negatives | AUC learned (random) | AUC equal (random) | Candidate pairs |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.948309 | 100.0% (30/30) | 50.0% (20/40) | 1.000000 | 1.000000 | 544 | 1.000000 | 1.000000 | 3198 |
| B | 0.922424 | 100.0% (30/30) | 25.0% (10/40) | 1.000000 | 1.000000 | 2647 | 1.000000 | 1.000000 | 5286 |
| C | 0.924302 | 100.0% (30/30) | 42.5% (17/40) | 0.755284 | 0.760108 | 143 | 0.909028 | 0.908947 | 1675 |

- A: blank 25% of email, phone. Columns kept: first_name, last_name, email, phone, address, city, zip_code, SSN. Excluded: customer_id, source_system.
- B: blank 50% of email, phone; typos in 30% of first_name, last_name, address. Columns kept: first_name, last_name, email, phone, address, city, zip_code, SSN. Excluded: customer_id, source_system.
- C: blank 50% of every column except the record id. Columns kept: first_name, last_name, email, phone, address, city, zip_code, SSN. Excluded: customer_id, source_system.

## Comparator baseline

Mean field_similarity with each column's own comparison method, over pairs where both values are non-null. Non-duplicate set: the same 5000 random non-duplicate pairs used above. Duplicate set: all true duplicate pairs. Difference = duplicate mean minus non-duplicate mean, smallest first.

| Dataset | Column | Method | Non-dup mean | n | Dup mean | n | Difference |
|---|---|---|---|---|---|---|---|
| products | member_id | text_bigram | 0.598400 | 5000 | 0.596675 | 401 | -0.001725 |
| products | member_part_number | bigram_jaccard | 0.504172 | 5000 | 0.721879 | 401 | 0.217706 |
| customers | email | email | 0.447125 | 4617 | 0.831782 | 2444 | 0.384658 |
| products | manufacturer_code | text_bigram | 0.473573 | 5000 | 1.000000 | 401 | 0.526427 |
| products | brand_code | text_bigram | 0.384860 | 5000 | 1.000000 | 401 | 0.615140 |
| customers | last_name | text_bigram | 0.024344 | 4822 | 0.859244 | 2577 | 0.834899 |
| customers | first_name | text_bigram | 0.031051 | 4806 | 0.866327 | 2576 | 0.835276 |
| customers | phone | text_bigram | 0.055772 | 3811 | 0.902933 | 2060 | 0.847161 |
| products | upc_ean | text_bigram | 0.057267 | 3873 | 0.941440 | 306 | 0.884173 |
| customers | SSN | text_bigram | 0.043269 | 4498 | 0.935746 | 2391 | 0.892478 |
| customers | city | text_bigram | 0.052009 | 5000 | 0.947792 | 2678 | 0.895783 |
| customers | address | text_bigram | 0.038567 | 5000 | 0.952165 | 2678 | 0.913598 |
| customers | zip_code | text_bigram | 0.023171 | 5000 | 0.947310 | 2678 | 0.924138 |
| products | manufacturer_part_number | text_bigram | 0.012299 | 5000 | 0.940834 | 401 | 0.928536 |
