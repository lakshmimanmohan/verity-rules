# Reliability diagnosis 2

Report only. Nothing in `verity_rules/` was changed. Seed 42. New = current code (HEAD 5712630). Old = commit b2436de (old uniqueness: distinct values over all rows, raw anchor-mean reliability), imported from a git worktree.

Definitions. **Learned reliability**: `reliability_score` from `learn_rules` of that code. **Ideal reliability**: `clip((dup mean - random mean) / (1 - random mean), 0, 1)` where dup mean is the mean `field_similarity` over every true duplicate pair with both values non-null, and random mean is the package's own random-pair mean (the 1000 random pairs `learn_rules` samples). **Anchor purity**: share of that column's anchor pairs that are true duplicates (anchors with relax off, so each min_agree level shows its own). With the old code the learned value is the raw anchor mean (no chance correction existed), so it is compared against the same chance-corrected ideal and is not expected to match it.

## Findings

- **Copy H2: Verity keeps all four pooled noise columns** (uniqueness 0.54 to 0.55, so not excluded as keys) but gives them almost no weight: 0.002241 in total, from reliability 0 to 0.0099. They do cost something elsewhere: with uniqueness above 0.5 they become candidate sources and add 10,831 candidate pairs, none true duplicates, so the true-duplicate share of candidates falls from 31.1% (Copy H) to 12.5%.
- **Best cutoff**: learned weights make 0 errors on original customers, 4 on Copy H and 3 on Copy H2 (pairs, hard negatives). Equal, uniqueness-only and v1 weights make 177 to 232 on Copy H and 217 to 388 on Copy H2. Original customers is 0 for all four.
- **Learned vs ideal reliability (new code)**: largest gap on any column is 0.0007 in customers, 0.0192 in Copy H (phone) and 0.0396 in Copy C (email). The old code overshoots ideal by 0.13 to 0.28 on email, phone, city and zip_code in Copy H and customers (and by 0.32 and 0.47 on email and phone in Copy C, where it had no candidates and fell back to 1.0), because it has no chance correction.
- **Ideal depends on the comparator**: email and phone in Copy C and Copy H are DESCRIPTIVE with fuzzy comparison under the old uniqueness (ideal 0.53 to 0.69) and IDENTIFIER with exact comparison under the new (ideal 0.15 to 0.21). The new uniqueness pushed them over 0.90.
- **Anchor purity**: at min_agree 2 every column is 98.5% to 100% pure in every dataset. At min_agree 1 purity is 25% to 52% in Copy H (the default run never uses it because every column has at least 270 anchors at level 2), 83% to 84% in customers (city 98%) and 89% to 99.7% in Copy C.
- **Candidates**: Copy C goes from 0 to 1675 candidates, 91.5% true duplicates. Copy H 30.8% to 31.1%. customers unchanged at 82.9%.
- **Copy C email/phone**: see section 5. Chance correction was a no-op because they became exact-match columns whose random-pair mean is 0.0.

## 1. Copy H2

Copy H2 = Copy H settings (blank 50% of email and phone; typos in 30% of first_name, last_name, address; scramble 60% of zip_code and city) with the four unique noise columns replaced by `--noise-pool-cols 2 --pool-size 3000`: two 8-letter-word columns and two 10-digit columns, each drawn with replacement from its own pool of 3000 random values. Built with: `python tools/degrade.py data/customers.csv data/customers_degraded_H2.csv --blank email,phone 0.5 --typo first_name,last_name,address 0.3 --scramble zip_code,city 0.6 --noise-pool-cols 2 --pool-size 3000`. Truth file unchanged.

Excluded: customer_id (primary_key+system_process_field); source_system (near_constant_field). Verity **keeps all four noise columns** (they are not unique enough to look like keys).

| Column | Semantic type | Comparison | Uniqueness | Null rate | Learned reliability | Ideal reliability | Composite weight |
|---|---|---|---|---|---|---|---|
| first_name | DESCRIPTIVE | text_character | 0.4878 | 0.0195 | 0.8334 | 0.8545 | 0.163114 |
| last_name | DESCRIPTIVE | text_character | 0.5718 | 0.0180 | 0.8503 | 0.8582 | 0.195405 |
| email | IDENTIFIER | exact | 0.9252 | 0.5222 | 0.1511 | 0.1488 | 0.027338 |
| phone | IDENTIFIER | exact | 0.9486 | 0.5620 | 0.1569 | 0.1785 | 0.026686 |
| address | STREET_ADDRESS | text_character | 0.8095 | 0.0000 | 0.9182 | 0.9473 | 0.304187 |
| city | DESCRIPTIVE | text_character | 0.4422 | 0.0000 | 0.1443 | 0.1563 | 0.026119 |
| zip_code | DESCRIPTIVE | text_character | 0.5202 | 0.0000 | 0.1217 | 0.1260 | 0.025911 |
| SSN | QUASI_IDENTIFIER | text_character | 0.6377 | 0.0548 | 0.9283 | 0.9637 | 0.229000 |
| noise_pool_word_1 | DESCRIPTIVE | text_character | 0.5395 | 0.0000 | 0.0000 | 0.0000 | 0.000000 |
| noise_pool_word_2 | DESCRIPTIVE | text_character | 0.5533 | 0.0000 | 0.0099 | 0.0055 | 0.002241 |
| noise_pool_digits_1 | DESCRIPTIVE | numeric_range | 0.5547 | 0.0000 | 0.0000 | 0.0000 | 0.000000 |
| noise_pool_digits_2 | DESCRIPTIVE | numeric_range | 0.5425 | 0.0000 | 0.0000 | 0.0028 | 0.000000 |

Weight on the four noise columns: 0.002241 in total. Separability: clear (gap 0.587729).

Candidate pairs: 18132, of which 2274 are true duplicates (12.5%). Pairs by source column (true duplicates): last_name: 2749 (643), email: 119 (117), phone: 71 (71), address: 609 (609), zip_code: 3026 (107), SSN: 727 (727), noise_pool_word_1: 2752 (0), noise_pool_word_2: 2653 (0), noise_pool_digits_1: 2628 (0), noise_pool_digits_2: 2798 (0). The four noise columns (uniqueness above 0.5) became candidate sources and contribute 10831 pairs, none of them true duplicates.


## 2. Best cutoff

# Best cutoff

### Best cutoff, customers

Positives: 2678 true duplicate pairs. Hard negatives: 549 candidate pairs that are not true duplicates (of 3209 candidates, 549 non-duplicate). Random negatives: 5000. Errors = missed duplicates + false merges at the best cutoff, in pairs.

| Weighting | AUC (hard) | Best-cutoff errors (hard) | missed (hard) | false merges (hard) | AUC (random) | Best-cutoff errors (random) | missed (random) | false merges (random) |
|---|---|---|---|---|---|---|---|---|
| learned | 1.000000 | 0 | 0 | 0 | 1.000000 | 0 | 0 | 0 |
| equal | 1.000000 | 0 | 0 | 0 | 1.000000 | 0 | 0 | 0 |
| v1 | 1.000000 | 0 | 0 | 0 | 1.000000 | 0 | 0 | 0 |
| uniqueness | 1.000000 | 0 | 0 | 0 | 1.000000 | 0 | 0 | 0 |

### Best cutoff, Copy H

Positives: 2678 true duplicate pairs. Hard negatives: 5000 candidate pairs that are not true duplicates (of 7301 candidates, 5027 non-duplicate; capped at 5000). Random negatives: 5000. Errors = missed duplicates + false merges at the best cutoff, in pairs.

| Weighting | AUC (hard) | Best-cutoff errors (hard) | missed (hard) | false merges (hard) | AUC (random) | Best-cutoff errors (random) | missed (random) | false merges (random) |
|---|---|---|---|---|---|---|---|---|
| learned | 0.999998 | 4 | 2 | 2 | 1.000000 | 0 | 0 | 0 |
| equal | 0.996265 | 177 | 119 | 58 | 0.999929 | 27 | 14 | 13 |
| v1 | 0.995483 | 197 | 71 | 126 | 0.999843 | 36 | 20 | 16 |
| uniqueness | 0.993465 | 232 | 177 | 55 | 0.999786 | 44 | 25 | 19 |

### Best cutoff, Copy H2

Positives: 2678 true duplicate pairs. Hard negatives: 5000 candidate pairs that are not true duplicates (of 18132 candidates, 15858 non-duplicate; capped at 5000). Random negatives: 5000. Errors = missed duplicates + false merges at the best cutoff, in pairs.

| Weighting | AUC (hard) | Best-cutoff errors (hard) | missed (hard) | false merges (hard) | AUC (random) | Best-cutoff errors (random) | missed (random) | false merges (random) |
|---|---|---|---|---|---|---|---|---|
| learned | 0.999998 | 3 | 2 | 1 | 1.000000 | 0 | 0 | 0 |
| equal | 0.986797 | 388 | 246 | 142 | 0.998682 | 108 | 70 | 38 |
| v1 | 0.995339 | 217 | 128 | 89 | 0.999215 | 72 | 37 | 35 |
| uniqueness | 0.987096 | 387 | 253 | 134 | 0.998411 | 125 | 84 | 41 |


## 3. Learned vs ideal reliability, anchors and purity

### New code (HEAD)

#### Copy C

2678 true duplicate pairs. Candidate pairs: 1675. Kept: first_name, last_name, email, phone, address, city, zip_code, SSN.

| Column | Uniqueness | Learned reliability | Ideal reliability | Learned - ideal | Anchors, min_agree 1 | Purity, 1 | Anchors, min_agree 2 | Purity, 2 | Level used |
|---|---|---|---|---|---|---|---|---|---|
| first_name | 0.3536 | 0.9525 | 0.9561 | -0.0036 | 423 | 88.7% | 142 | 100.0% | 2 |
| last_name | 0.4753 | 0.9584 | 0.9474 | 0.0111 | 407 | 91.9% | 143 | 100.0% | 2 |
| email | 0.9201 | 0.1231 | 0.1627 | -0.0396 | 383 | 90.1% | 130 | 100.0% | 2 |
| phone | 0.9348 | 0.2277 | 0.2097 | 0.0180 | 328 | 94.5% | 101 | 100.0% | 2 |
| address | 0.7920 | 0.9837 | 0.9842 | -0.0005 | 388 | 91.2% | 125 | 100.0% | 2 |
| city | 0.7340 | 0.9771 | 0.9836 | -0.0064 | 370 | 99.7% | 124 | 100.0% | 2 |
| zip_code | 0.8255 | 0.8476 | 0.8441 | 0.0035 | 393 | 91.9% | 101 | 100.0% | 2 |
| SSN | 0.7986 | 0.9624 | 0.9613 | 0.0011 | 348 | 88.8% | 94 | 100.0% | 2 |

#### Copy H

2678 true duplicate pairs. Candidate pairs: 7301. Kept: first_name, last_name, email, phone, address, city, zip_code, SSN.

| Column | Uniqueness | Learned reliability | Ideal reliability | Learned - ideal | Anchors, min_agree 1 | Purity, 1 | Anchors, min_agree 2 | Purity, 2 | Level used |
|---|---|---|---|---|---|---|---|---|---|
| first_name | 0.4878 | 0.8473 | 0.8545 | -0.0072 | 7019 | 31.2% | 1406 | 99.7% | 2 |
| last_name | 0.5718 | 0.8602 | 0.8582 | 0.0020 | 4899 | 42.8% | 1020 | 100.0% | 2 |
| email | 0.9252 | 0.1536 | 0.1488 | 0.0048 | 1655 | 31.9% | 306 | 99.7% | 2 |
| phone | 0.9486 | 0.1593 | 0.1785 | -0.0192 | 1408 | 32.8% | 270 | 98.5% | 2 |
| address | 0.8095 | 0.9406 | 0.9473 | -0.0066 | 7148 | 29.7% | 976 | 99.6% | 2 |
| city | 0.4422 | 0.1467 | 0.1563 | -0.0096 | 7301 | 31.1% | 1454 | 99.7% | 2 |
| zip_code | 0.5202 | 0.1234 | 0.1260 | -0.0026 | 4349 | 51.5% | 1340 | 100.0% | 2 |
| SSN | 0.6377 | 0.9608 | 0.9637 | -0.0029 | 6045 | 26.2% | 572 | 99.3% | 2 |

#### customers

2678 true duplicate pairs. Candidate pairs: 3209. Kept: first_name, last_name, email, phone, address, city, zip_code, SSN.

| Column | Uniqueness | Learned reliability | Ideal reliability | Learned - ideal | Anchors, min_agree 1 | Purity, 1 | Anchors, min_agree 2 | Purity, 2 | Level used |
|---|---|---|---|---|---|---|---|---|---|
| first_name | 0.2721 | 0.9553 | 0.9554 | -0.0002 | 3089 | 82.8% | 2524 | 100.0% | 2 |
| last_name | 0.3607 | 0.9523 | 0.9519 | 0.0004 | 3090 | 82.8% | 2523 | 100.0% | 2 |
| email | 0.8576 | 0.6969 | 0.6977 | -0.0007 | 2921 | 83.0% | 2373 | 100.0% | 2 |
| phone | 0.8865 | 0.5333 | 0.5337 | -0.0004 | 2452 | 83.4% | 1991 | 100.0% | 2 |
| address | 0.6332 | 0.9830 | 0.9828 | 0.0002 | 3204 | 82.9% | 2513 | 100.0% | 2 |
| city | 0.5577 | 0.9844 | 0.9848 | -0.0003 | 2703 | 98.1% | 2476 | 100.0% | 2 |
| zip_code | 0.6925 | 0.8387 | 0.8388 | -0.0001 | 3156 | 83.9% | 2438 | 100.0% | 2 |
| SSN | 0.6377 | 0.9632 | 0.9637 | -0.0005 | 2871 | 82.5% | 2262 | 100.0% | 2 |

### Old code (b2436de)

#### Copy C

2678 true duplicate pairs. Candidate pairs: 0. Kept: first_name, last_name, email, phone, address, city, zip_code, SSN. No candidate pairs, so no anchors; every column's learned reliability is the fallback 1.0.

| Column | Uniqueness | Learned reliability | Ideal reliability | Learned - ideal | Anchors, min_agree 1 | Purity, 1 | Anchors, min_agree 2 | Purity, 2 | Level used |
|---|---|---|---|---|---|---|---|---|---|
| first_name | 0.1732 | 1.0000 | 0.9561 | 0.0439 | 0 | n/a | 0 | n/a | 1 |
| last_name | 0.2338 | 1.0000 | 0.9474 | 0.0526 | 0 | n/a | 0 | n/a | 1 |
| email | 0.4405 | 1.0000 | 0.6803 | 0.3197 | 0 | n/a | 0 | n/a | 1 |
| phone | 0.4125 | 1.0000 | 0.5329 | 0.4671 | 0 | n/a | 0 | n/a | 1 |
| address | 0.3962 | 1.0000 | 0.9842 | 0.0158 | 0 | n/a | 0 | n/a | 1 |
| city | 0.3673 | 1.0000 | 0.9836 | 0.0164 | 0 | n/a | 0 | n/a | 1 |
| zip_code | 0.4130 | 1.0000 | 0.8441 | 0.1559 | 0 | n/a | 0 | n/a | 1 |
| SSN | 0.3780 | 1.0000 | 0.9613 | 0.0387 | 0 | n/a | 0 | n/a | 1 |

#### Copy H

2678 true duplicate pairs. Candidate pairs: 7265. Kept: first_name, last_name, email, phone, address, city, zip_code, SSN.

| Column | Uniqueness | Learned reliability | Ideal reliability | Learned - ideal | Anchors, min_agree 1 | Purity, 1 | Anchors, min_agree 2 | Purity, 2 | Level used |
|---|---|---|---|---|---|---|---|---|---|
| first_name | 0.4785 | 0.8824 | 0.8545 | 0.0279 | 6986 | 30.9% | 1270 | 99.7% | 2 |
| last_name | 0.5617 | 0.8880 | 0.8582 | 0.0298 | 4852 | 42.3% | 838 | 100.0% | 2 |
| email | 0.4422 | 0.8299 | 0.6897 | 0.1402 | 1641 | 31.3% | 283 | 99.6% | 2 |
| phone | 0.4158 | 0.7286 | 0.5206 | 0.2080 | 1400 | 32.4% | 253 | 98.4% | 2 |
| address | 0.8095 | 0.9626 | 0.9473 | 0.0153 | 7089 | 29.1% | 788 | 99.5% | 2 |
| city | 0.4422 | 0.4358 | 0.1563 | 0.2795 | 7265 | 30.8% | 1312 | 99.7% | 2 |
| zip_code | 0.5202 | 0.3552 | 0.1260 | 0.2292 | 4311 | 51.1% | 1184 | 100.0% | 2 |
| SSN | 0.6030 | 0.9798 | 0.9637 | 0.0161 | 5926 | 24.7% | 441 | 99.1% | 2 |

#### customers

2678 true duplicate pairs. Candidate pairs: 3209. Kept: first_name, last_name, email, phone, address, city, zip_code, SSN.

| Column | Uniqueness | Learned reliability | Ideal reliability | Learned - ideal | Anchors, min_agree 1 | Purity, 1 | Anchors, min_agree 2 | Purity, 2 | Level used |
|---|---|---|---|---|---|---|---|---|---|
| first_name | 0.2670 | 0.9655 | 0.9554 | 0.0100 | 3089 | 82.8% | 2524 | 100.0% | 2 |
| last_name | 0.3545 | 0.9620 | 0.9519 | 0.0101 | 3090 | 82.8% | 2523 | 100.0% | 2 |
| email | 0.8225 | 0.8314 | 0.6977 | 0.1337 | 2921 | 83.0% | 2373 | 100.0% | 2 |
| phone | 0.7775 | 0.7395 | 0.5337 | 0.2058 | 2452 | 83.4% | 1991 | 100.0% | 2 |
| address | 0.6332 | 0.9894 | 0.9828 | 0.0066 | 3204 | 82.9% | 2513 | 100.0% | 2 |
| city | 0.5577 | 0.9896 | 0.9848 | 0.0049 | 2703 | 98.1% | 2476 | 100.0% | 2 |
| zip_code | 0.6925 | 0.8815 | 0.8388 | 0.0427 | 3156 | 83.9% | 2438 | 100.0% | 2 |
| SSN | 0.6030 | 0.9814 | 0.9637 | 0.0177 | 2871 | 82.5% | 2262 | 100.0% | 2 |

## 4. Candidate pairs: share that are true duplicates

| Dataset | Old: candidate pairs | Old: true duplicates | Old: share | New: candidate pairs | New: true duplicates | New: share |
|---|---|---|---|---|---|---|
| Copy C | 0 | 0 | n/a | 1675 | 1532 | 91.5% |
| Copy H | 7265 | 2240 | 30.8% | 7301 | 2274 | 31.1% |
| customers | 3209 | 2660 | 82.9% | 3209 | 2660 | 82.9% |

Candidate source columns (uniqueness >= 0.5). Pairs counted under the column whose group produced them first (first column in profile order wins when a pair shares values in several). Format: pairs (true duplicates).

| Dataset | Code | Columns qualifying as sources (uniqueness) | Pairs by source |
|---|---|---|---|
| Copy C | old | none | none |
| Copy C | new | SSN (0.799), address (0.792), city (0.734), email (0.920), phone (0.935), zip_code (0.826) | SSN: 223 (223), address: 434 (434), city: 507 (378), email: 161 (160), phone: 107 (107), zip_code: 243 (230) |
| Copy H | old | SSN (0.603), address (0.809), last_name (0.562), zip_code (0.520) | SSN: 803 (803), address: 678 (678), last_name: 2749 (643), zip_code: 3035 (116) |
| Copy H | new | SSN (0.638), address (0.809), email (0.925), last_name (0.572), phone (0.949), zip_code (0.520) | SSN: 727 (727), address: 609 (609), email: 119 (117), last_name: 2749 (643), phone: 71 (71), zip_code: 3026 (107) |
| customers | old | SSN (0.603), address (0.633), city (0.558), email (0.823), phone (0.777), zip_code (0.693) | SSN: 32 (32), address: 1227 (1227), city: 924 (427), email: 604 (594), phone: 314 (314), zip_code: 108 (66) |
| customers | new | SSN (0.638), address (0.633), city (0.558), email (0.858), phone (0.887), zip_code (0.693) | SSN: 32 (32), address: 1227 (1227), city: 924 (427), email: 604 (594), phone: 314 (314), zip_code: 108 (66) |

## 5. Copy C: why email and phone reliability do not change with change 2

Copy C email and phone reliability are 0.123077 and 0.227723 both before and after change 2.

Path (new code):
1. Change 1 (`column_uniqueness_and_nulls`, `rule_learner.py:164`) divides distinct values by non-null count, so Copy C email and phone uniqueness rise from 0.4405 / 0.4125 to 0.9201 / 0.9348. Both are above 0.90, so `infer_semantic_type` (`rule_learner.py:221-222` for email, `:241` for phone) returns IDENTIFIER and `field_similarity` (`rule_learner.py:421`) returns 1.0 for equal values and 0.0 otherwise (`comparison_method` = exact).
2. `learn_rules` calls `learn_reliability_v2(..., random_pairs=pairs)` (`core.py:233`). Inside it, the random-pair block (`rule_learner.py:721-735`) averages `field_similarity` over the 1000 random pairs where both values are non-null: 217 pairs for email and 194 for phone. None of them are equal, so `random_mean` is exactly 0.0 for both.
3. The correction at `rule_learner.py:743`, `(anchor_mean - r) / (1.0 - r)` with r = 0.0, equals anchor_mean. The ceiling branch (`rule_learner.py:738`, r >= 0.999) and the all-zero fallback were not taken, and there are no relaxation notes.
So reliability equals the raw anchor mean (130 anchors for email, 101 for phone, min_agree 2): 0.123077 = 16/130 and 0.227723 = 23/101.

Checked by running `learn_reliability_v2` on the same candidates with `random_pairs=None` (correction off) and with the 1000 random pairs: identical output for both columns. Every other column has a nonzero random mean (0.17 to 0.51) and moves by 0.004 to 0.04.
