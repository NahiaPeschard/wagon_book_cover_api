x.DEFAULT_GOAL := default

reinstall_package:
	@pip uninstall -y book_cover_api || :
	@pip install -e .

run_api:
	uvicorn book_cover_api.api.book_api:app --reload
