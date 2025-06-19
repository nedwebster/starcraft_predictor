build_datasets:
	uv run python scripts/build_datasets.py

run_backend:
	uv run fastapi dev

run_frontend:
	npm run dev --prefix frontend