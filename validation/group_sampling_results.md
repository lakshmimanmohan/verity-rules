# Group-aware row sampling

Seed 42. Report only; nothing tuned. `groups` keeps rows that share a normalized value in a distinctive column together (groups of 2 to 50 rows); `random` draws rows uniformly.

## customers.csv (4,000 rows), small caps

Reference: weights from all 4,000 rows. Max weight difference is the largest absolute difference in composite_weight over the columns kept in both runs; rank correlation is Spearman over the same columns.

| --max-rows | Mode | Rows analysed | Groups kept | Candidate pairs | Max weight diff | Rank correlation |
|---|---|---|---|---|---|---|
| 400 | groups | 400 | 130 | 667 | 0.0385 | 1.000 |
| 400 | random | 400 | 0 | 429 | 0.1295 | -0.357 |
| 1000 | groups | 1000 | 331 | 745 | 0.0254 | 1.000 |
| 1000 | random | 1000 | 0 | 801 | 0.0301 | 0.643 |
| 2000 | groups | 2000 | 642 | 1582 | 0.0129 | 1.000 |
| 2000 | random | 2000 | 0 | 786 | 0.0182 | 0.857 |


## customers_100k.csv (100,000 rows, 50,000 entities)

Each run is `verity learn` in a fresh process, so time includes Python start and imports. Peak memory is the process peak working set. Weight difference and rank correlation compare each run with the full run. The last column is the share of true duplicate pairs (from the truth file, read here only) whose two rows are both in the analysed rows.

| Run | Rows analysed | Candidate pairs | Seconds | Peak memory (MB) | Max weight diff | Rank correlation | True duplicate pairs kept |
|---|---|---|---|---|---|---|---|
| full (--max-rows 100000) | 100000 | 20000 | 91.8 | 343 | 0.0000 | 1.000 | 66,729 of 66,729 (100.0%) |
| default cap (20,000), groups | 20000 | 19614 | 28.0 | 263 | 0.0263 | 0.976 | 13,321 of 66,729 (20.0%) |
| default cap (20,000), random | 20000 | 14341 | 25.0 | 235 | 0.0404 | 0.833 | 2,654 of 66,729 (4.0%) |

