.PHONY:  all down copy-schemas build  up generate-clients test-worker test-it-tests create-release


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
	docker-compose exec gpu-workers pytest tests/$(TEST_PATH) -vs

# make test-it-tests TEST_PATH=images
# make test-it-tests TEST_PATH=videos
# make test-it-tests TEST_PATH=texts
test-it-tests: generate-clients
	cd it_tests && pytest $(TEST_PATH) -vs
	cd ..

VERSION ?= 0.3.0
PROJECT_NAME ?= deferred-diffusion
create-release: build
# Define variables
	$(eval USERNAME=joegaffney)
	$(eval REPO=deferred-diffusion)
# Create release directory with combined project-version name
	if not exist releases mkdir releases
	if exist releases\$(VERSION)\$(PROJECT_NAME) rmdir /S /Q releases\$(VERSION)\$(PROJECT_NAME)
	mkdir releases\$(VERSION)\$(PROJECT_NAME)

# Tag images with version (this creates new tags without removing latest tags)
	docker tag deferred-diffusion-api:latest $(USERNAME)/$(REPO):api-$(VERSION)
	docker tag deferred-diffusion-workers:latest $(USERNAME)/$(REPO):worker-$(VERSION)
# Push images
	docker push $(USERNAME)/$(REPO):api-$(VERSION)
	docker push $(USERNAME)/$(REPO):worker-$(VERSION)

# Copy deployment docker-compose.yml to release folder
	copy docker-compose.release.yml releases\$(VERSION)\$(PROJECT_NAME)\docker-compose.yml
	copy README.md releases\$(VERSION)\$(PROJECT_NAME)\README.md

# Update docker-compose.yml to use versioned images
	powershell -Command "(Get-Content releases\$(VERSION)\$(PROJECT_NAME)\docker-compose.yml) -replace 'deferred-diffusion-api:latest', '$(USERNAME)/$(REPO):api-$(VERSION)' -replace 'deferred-diffusion-workers:latest', '$(USERNAME)/$(REPO):worker-$(VERSION)' | Set-Content releases\$(VERSION)\$(PROJECT_NAME)\docker-compose.yml"

# Copy directories with exclusions
	echo __pycache__ > exclude_patterns.txt
	echo backup >> exclude_patterns.txt
	xcopy /E /I /Y /EXCLUDE:exclude_patterns.txt nuke releases\$(VERSION)\$(PROJECT_NAME)\nuke
	xcopy /E /I /Y /EXCLUDE:exclude_patterns.txt hda releases\$(VERSION)\$(PROJECT_NAME)\hda
	del exclude_patterns.txt
