---
name: quality-of-earnings
description: Checklist for judging whether reported earnings are backed by cash. Use when assessing receivables, cash conversion, accruals, or capex for any company.
---

# Quality-of-earnings checklist

Work through these in order. Report each as clean, flagged, or not assessable.

1. **Receivables vs. revenue.** Receivables growing faster than revenue by >10pp can mean
   looser terms, channel stuffing, or collection problems. Check the filing for changed
   payment terms and customer concentration.
2. **DSO trend.** A rise of >5 days year over year warrants an explanation in MD&A.
3. **Cash conversion (CFO / net income).** Persistently <0.9 means earnings are not turning
   into cash. One weak year with a stated cause (for example a one-off tax payment) is
   less concerning than a trend.
4. **Accruals ratio ((net income − CFO) / average assets).** Above ~5% is a warning sign.
5. **Capex vs. depreciation.** Below 1.0x for years may mean under-investment flattering
   margins; well above 1.0x may mean growth spend. Say which and why.
6. **One-offs.** Look for gains, restructuring, or reclassifications in MD&A and notes.

## Rules
- Use `compute_ratios` for the numbers. Do not recompute by hand.
- A flag is a question, not a verdict. State what the filing says about the cause.
- Run `run_python` only for analysis the tool does not cover (for example a custom
  sensitivity). Its files: `facts.json` when a ticker is passed.
