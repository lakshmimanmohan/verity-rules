# Known limits (v0.1.1)
Source: validation/*.md. All data below is synthetic.

Tested (synthetic customers 4,000 rows, products 500 rows, degraded copies, customers 20k and 100k):
- Weights are stable across seeds 1 to 5 (range under 0.003) and rank order is unchanged.
- 30 of 30 likely_match and 30 of 30 likely_non_match pairs were true on customers and products.
- On customers, learned weights made 0 errors at the best cutoff; on the degraded copies H and H2, 2 and 1.
- Output is identical on repeat runs with the same seed.

Not tested or weak:
- No real customer data, no other domains or languages, no embeddings run.
- Small files: random 400-row subsets of a 4,000-row file moved weights by up to 0.16 and group-aware ones by 0.04; no genuine small real file has been tested; products (500 rows) moved up to 0.04 on 250-row random subsets. 50% subsets of customers moved weights by up to 0.006.
- Above --max-rows (20,000) only 20% of true duplicate pairs stay with groups sampling (4% random) on the 100k file.
- Expected pair coverage cutoff of 0.3 is a rule of thumb from synthetic tests.
- Products: no hard negatives existed, so the hard-negative AUC checks were not evaluable.
- 18 of 2,678 customer duplicates and 70 of 401 product duplicates had no shared value (not candidates).
- Unclear pairs: 37.5% true duplicates on customers, 100% on products; do not treat as labels.
- Comparator choice (v4) was tuned on these same datasets.
- likely_non_match pairs are random pairs, not hard negatives. Weights depend on the seed only through the random pair sample.
- With half of every column blank (Copy C), hard negatives score about as high as duplicates (AUC 0.76).
- member_part_number has the third-largest products weight, but its duplicate and random similarities overlap.
- Weighting under comparator v4 (validation/weighting_v4.md): learned weights tied equal and uniqueness-only weights, within two errors, on customers, Copy C and Copy C25.
  They beat both on Copy H and H2, where columns were corrupted or noise was added (2 and 1 errors, against 37 and 97 for equal and 15 and 45 for uniqueness-only).
- Hosted action, skill text and MCP server are not updated and not validated.
