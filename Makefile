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
# make test-worker TEST_PATH=videos
# make test-worker TEST_PATH=texts
# make test-worker TEST_PATH=images
# make test-worker TEST_PATH=images/test_text_to_image.py
# make test-worker TEST_PATH=images/test_image_to_image.py
# make test-worker TEST_PATH=images/test_external_text_to_image.py
# make test-worker TEST_PATH=images/test_external_image_to_image.py
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

RELEASE_VERSION ?= 0.1.0
RELEASE_PROJECT_NAME ?= deferred-diffusion
create-release: build
# Create release directory with combined project-version name
	if not exist releases mkdir releases
	if exist releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME) rmdir /S /Q releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)
	mkdir releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)

# Tag images with version (this creates new tags without removing latest tags)
	docker tag deferred-diffusion-api:latest deferred-diffusion-api:$(RELEASE_VERSION)
	docker tag deferred-diffusion-workers:latest deferred-diffusion-workers:$(RELEASE_VERSION)

# Save Docker images with latest tag
	docker save -o releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)\deferred-diffusion-api.tar deferred-diffusion-api:${RELEASE_VERSION}
	docker save -o releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)\deferred-diffusion-workers.tar deferred-diffusion-workers:${RELEASE_VERSION}

# Copy deployment docker-compose.yml to release folder
	copy docker-compose.release.yml releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)\docker-compose.yml
	copy README.md releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)\README.md

# Update docker-compose.yml to use versioned images
	powershell -Command "(Get-Content releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)\docker-compose.yml) -replace 'deferred-diffusion-api:latest', 'deferred-diffusion-api:$(RELEASE_VERSION)' -replace 'deferred-diffusion-workers:latest', 'deferred-diffusion-workers:$(RELEASE_VERSION)' | Set-Content releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)\docker-compose.yml"

# Copy directories with exclusions
	echo __pycache__ > exclude_patterns.txt
	echo backup >> exclude_patterns.txt
	xcopy /E /I /Y /EXCLUDE:exclude_patterns.txt nuke releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)\nuke
	xcopy /E /I /Y /EXCLUDE:exclude_patterns.txt hda releases\$(RELEASE_VERSION)\$(RELEASE_PROJECT_NAME)\hda
	del exclude_patterns.txt
