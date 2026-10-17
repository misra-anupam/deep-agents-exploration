.PHONY: test eval eval-live phase1 demo clean-data

test:            ## offline unit + integration tests (no API key needed)
	uv run pytest -q

eval:            ## offline: verify the scorer separates good from bad notes
	uv run python -m equity_desk.evals samples

eval-live:       ## real model on the golden event; needs OPENROUTER_API_KEY
	uv run python -m equity_desk.evals live

demo:            ## interactive run on the synthetic company; needs OPENROUTER_API_KEY
	uv run equity-desk run ACME --period FY2024

clean-data:      ## wipe local checkpoints, store, broker, audit log, filing cache
	rm -rf data/*.sqlite data/audit.jsonl data/filings_cache
