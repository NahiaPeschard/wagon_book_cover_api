x.DEFAULT_GOAL := default

reinstall_package:
	@pip uninstall -y book_cover_api || :
	@pip install -e .

run_api:
	uvicorn book_cover_api.api.book_api:app --reload

docker_build:
	docker build -t $GCR_REGION/$GCP_PROJECT/$GCR_IMAGE:prod .

docker_push:
	docker push $GCR_REGION/$GCP_PROJECT/$GCR_IMAGE:prod

api_deploy:
	gcloud run deploy --image $GCR_REGION/$GCP_PROJECT/$GCR_IMAGE:prod --memory $GCR_MEMORY --region $GCP_REGION --env-vars-file .env.yaml
