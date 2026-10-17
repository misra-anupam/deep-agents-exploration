---
name: earnings-note-format
description: House template and citation rules for earnings research notes. Use when drafting or revising any research note under /reports/.
---

# Earnings note format

Write notes to `/reports/<TICKER>-<PERIOD>.md` (for example `/reports/ACME-FY2024.md`).

## Structure
1. **Headline** — one sentence: what changed and the stance (Positive / Neutral / Negative).
2. **Results vs. prior year** — revenue, gross margin, net income, operating cash flow.
3. **Quality of earnings** — receivables, cash conversion, one-offs. State what you checked even if clean.
4. **Risks** — drawn from the filing's risk factors and MD&A, not generic boilerplate.
5. **What would change our mind** — two or three observable, falsifiable signals.
6. **Sources** — list every filing path and XBRL accession used.

## Citation rules
- Every number carries a citation in brackets: `[10-K 0009999999-25-000001, Item 7]` for text,
  `[XBRL Revenues FY2024, accn 0009999999-25-000001]` for facts.
- Prefer XBRL values for figures; use filing text for explanations.
- If you cannot source a number, leave it out. Never estimate silently.

## Language
- Describe what the filing reports. Do not promise outcomes ("will", "guaranteed").
- End every note with: `Research aid only. Not investment advice.`
