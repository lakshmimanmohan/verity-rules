# Known limits (v0.1.0)
Source: validation/*.md. All data below is synthetic.

Tested (synthetic customers 4,000 rows, products 500 rows, degraded copies, customers 20k and 100k):
- Weights are stable across seeds 1 to 5 (range under 0.003) and rank order is unchanged.
- 30 of 30 likely_match and 30 of 30 likely_non_match pairs were true on customers and products.
- On customers, learned weights made 0 errors at the best cutoff; on the degraded copies H and H2, 2 and 1.
- Output is identical on repeat runs with the same seed.

Not tested or weak:
- No real customer data, no other domains or languages, no embeddings run.
- 50% row subsets move weights by up to 0.006 on customers; products (250 rows) by up to 0.04. Top 3 columns held, order did not always.
- Below about 1,000 rows weights are unstable (a 400-row subset moved weights by up to 0.16).
- Above --max-rows (20,000) only 20% of true duplicate pairs stay with groups sampling (4% random) on the 100k file.
- Expected pair coverage cutoff of 0.3 is a rule of thumb from synthetic tests.
- Products: no hard negatives existed, so the hard-negative AUC checks were not evaluable.
- 18 of 2,678 customer duplicates and 70 of 401 product duplicates had no shared value (not candidates).
- Unclear pairs: 37.5% true duplicates on customers, 100% on products; do not treat as labels.
- Comparator choice (v4) was tuned on these same datasets.
- Hosted action, skill text and MCP server are not updated and not validated.
