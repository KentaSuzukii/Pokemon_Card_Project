# Makefile for building, running, and deploying the Streamlit app

PROJECT_NAME=pokemon-card-app
IMAGE_NAME=gcr.io/pokemon-card-app-1983/$(PROJECT_NAME)

# Build the Docker image
build:
	docker build -t $(PROJECT_NAME) .

build-no-cache:
	docker build --no-cache -t $(IMAGE_NAME) .

# Run the container locally
run:
	docker run -p 8501:8501 $(PROJECT_NAME)

# Tag and push the image to Google Container Registry (GCR)
push:
	docker tag $(PROJECT_NAME) $(IMAGE_NAME)
	docker push $(IMAGE_NAME)

# Deploy to Google Cloud Run
deploy:
	gcloud run deploy $(PROJECT_NAME) \
		--image $(IMAGE_NAME) \
		--platform managed \
		--region asia-northeast1 \
		--allow-unauthenticated \

# Clean up Docker images
clean:
	docker rmi -f $(PROJECT_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true
