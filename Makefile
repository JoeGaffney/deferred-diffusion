.PHONY: all build down up generate-clients test-texts test-images test-it-tests test-worker tag-and-push

# Default target
all: generate-clients

# Docker management
down:
	docker-compose down

build: down
	set DOCKER_BUILDKIT=1 
	docker-compose build

up: build
	docker-compose up -d

# API Client generation
generate-clients: up
	openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path hda/python/generated --overwrite
	openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path nuke/python/generated --overwrite
	openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path it_tests/generated --overwrite

# Test texts modules
test-texts: generate-clients
	docker-compose exec workers pytest tests/texts

# Test images modules
test-images: generate-clients
	docker-compose exec workers pytest tests/images

test-it-tests: generate-clients
	cd it_tests && pytest -vs
	cd ..

TEST_PATH ?= images/models/test_sdxl.py
# make test-worker TEST_PATH=images/models/test_sdxl.py
# make test-worker TEST_PATH=images/models/test_hi_dream.py
# make test-worker TEST_PATH=images/models/test_flux.py
test-worker: up
	docker-compose exec workers pytest tests/$(TEST_PATH) -vs

# Tag and push Docker images to GitHub Container Registry
tag-and-push:
# Define variables
	$(eval USERNAME=joegaffney)
	$(eval REPO=deferred-diffusion)
	$(eval VERSION=latest)
# Login to Docker Hub
	docker login
# Tag images with different tags in the same repository
	docker tag deferred-diffusion-api:latest $(USERNAME)/$(REPO):api-$(VERSION)
	docker tag deferred-diffusion-workers:latest $(USERNAME)/$(REPO):worker-$(VERSION)
# Push images
	docker push $(USERNAME)/$(REPO):api-$(VERSION)
	docker push $(USERNAME)/$(REPO):worker-$(VERSION)

