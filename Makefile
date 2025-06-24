.PHONY:  all down copy-schemas build  up generate-clients test-worker test-it-tests  tag-and-push


# Default target
all: generate-clients

# Docker management
down:
	docker-compose down

# atm we align by copying schemas from the api to the workers
# this is a temporary solution until we have a better way to manage schemas
copy-schemas:
	copy api\images\schemas.py workers\images\schemas.py
	copy api\texts\schemas.py workers\texts\schemas.py
	copy api\videos\schemas.py workers\videos\schemas.py

build: down copy-schemas
	docker-compose build

up: build
	docker-compose up -d

# API Client generation
generate-clients: up
	openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path hda/python/generated --overwrite
	openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path nuke/python/generated --overwrite
	openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path it_tests/generated --overwrite


TEST_PATH ?= images
# make test-worker TEST_PATH=images
# make test-worker TEST_PATH=videos
# make test-worker TEST_PATH=texts
# make test-worker TEST_PATH=images/models/test_sdxl.py
# make test-worker TEST_PATH=images/models/test_hi_dream.py
# make test-worker TEST_PATH=images/models/test_flux.py
# make test-worker TEST_PATH=images/external_models/test_flux_kontext.py
test-worker: up
	docker-compose exec workers pytest tests/$(TEST_PATH) -vs

# make test-it-tests TEST_PATH=images
# make test-it-tests TEST_PATH=videos
# make test-it-tests TEST_PATH=texts
test-it-tests: generate-clients
	cd it_tests && pytest $(TEST_PATH) -vs
	cd ..

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

