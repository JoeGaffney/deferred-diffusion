.PHONY:  all down copy-schemas build  up generate-clients test-worker test-it-tests create-release

VERSION ?= latest
PROJECT_NAME ?= deferred-diffusion
REPO ?= deferred-diffusion
REPO_USERNAME ?= joegaffney
TEST_PATH ?= images/models


# Default target
all: generate-clients

# Docker management
down:
	docker compose down

# atm we align by copying schemas from the api to the workers
# this is a temporary solution until we have a better way to manage schemas
copy-schemas:
ifeq ($(OS),Windows_NT)
	copy api\images\schemas.py workers\images\schemas.py
	copy api\texts\schemas.py workers\texts\schemas.py
	copy api\videos\schemas.py workers\videos\schemas.py
else
	cp api/images/schemas.py workers/images/schemas.py
	cp api/texts/schemas.py workers/texts/schemas.py
	cp api/videos/schemas.py workers/videos/schemas.py
endif

build: down copy-schemas
	docker compose build

up: build
	docker compose up -d

up-it-tests: down copy-schemas
	docker compose -f docker-compose.it-tests.yml build
	docker compose -f docker-compose.it-tests.yml up -d

# Generate OpenAPI spec file
generate-openapi-spec: up
	curl -o clients/openapi.json http://127.0.0.1:5000/openapi.json || powershell -Command "Invoke-WebRequest -Uri 'http://127.0.0.1:5000/openapi.json' -OutFile 'clients/openapi.json'"
	@echo OpenAPI spec saved to clients/openapi.json

# API Client generation
generate-clients-raw: 
	openapi-python-client generate --path clients/openapi.json --output-path clients/houdini/python/generated --overwrite
	openapi-python-client generate --path clients/openapi.json --output-path clients/nuke/python/generated --overwrite
	openapi-python-client generate --path clients/openapi.json --output-path clients/it_tests/generated --overwrite


# API Client generation
generate-clients: generate-openapi-spec generate-clients-raw
	@echo "API clients generated successfully."


# Example test commands:
# make test-worker TEST_PATH=images
# make test-worker TEST_PATH=images/models/test_flux.py
# make test-worker TEST_PATH=images/external_models/test_replicate.py
# make test-worker TEST_PATH=texts
# make test-worker TEST_PATH=videos
# make test-worker TEST_PATH=videos/models/test_wan.py
# make test-worker TEST_PATH=videos/external_models/test_replicate.py
test-worker: up
	docker compose exec gpu-workers pytest tests/$(TEST_PATH) -vs

test-worker-basic: up
	docker compose exec gpu-workers pytest -m "basic" -vs

# make it-tests TEST_PATH=test_image_local.py
it-tests: generate-clients
	cd clients/it_tests && pytest tests/$(TEST_PATH) -vs
	cd ../..

it-tests-local: generate-clients
	cd clients/it_tests && pytest -m "local" -vs
	cd ../..

it-tests-external: generate-clients
	cd clients/it_tests && pytest -m "external" -vs
	cd ../..

it-tests-external-ci:
	cd clients/it_tests && pytest -m "external" -vs
	cd ../..

tag-and-push: build
# Tag images with version (this creates new tags without removing latest tags)
	docker tag deferred-diffusion-api:latest $(REPO_USERNAME)/$(REPO):api-$(VERSION)
	docker tag deferred-diffusion-workers:latest $(REPO_USERNAME)/$(REPO):worker-$(VERSION)
# Push images
	docker push $(REPO_USERNAME)/$(REPO):api-$(VERSION)
	docker push $(REPO_USERNAME)/$(REPO):worker-$(VERSION)

create-client-release: generate-clients-raw
# Create release directory with combined project-version name
	if not exist releases mkdir releases
	if exist releases\$(VERSION)\$(PROJECT_NAME) rmdir /S /Q releases\$(VERSION)\$(PROJECT_NAME)
	mkdir releases\$(VERSION)\$(PROJECT_NAME)

# Copy deployment docker-compose.yml to release folder
	copy docker-compose.release.yml releases\$(VERSION)\$(PROJECT_NAME)\docker-compose.yml
	copy README.md releases\$(VERSION)\$(PROJECT_NAME)\README.md

# Update docker-compose.yml to use versioned images
	powershell -Command "(Get-Content releases\$(VERSION)\$(PROJECT_NAME)\docker-compose.yml) -replace 'deferred-diffusion-api:latest', '$(REPO_USERNAME)/$(REPO):api-$(VERSION)' -replace 'deferred-diffusion-workers:latest', '$(REPO_USERNAME)/$(REPO):worker-$(VERSION)' | Set-Content releases\$(VERSION)\$(PROJECT_NAME)\docker-compose.yml"

# Copy directories with exclusions
	xcopy /E /I /Y clients releases\$(VERSION)\$(PROJECT_NAME)\clients

# Create release archive
	cd releases\$(VERSION) && tar -czf $(PROJECT_NAME)-$(VERSION).tar.gz $(PROJECT_NAME)
	@echo Release files created in releases\$(VERSION)\$(PROJECT_NAME)
	@echo Archive created: releases\$(VERSION)\$(PROJECT_NAME)-$(VERSION).tar.gz

create-client-release-linux: generate-clients-raw
# Create release directory with combined project-version name
	mkdir -p releases/$(VERSION)/$(PROJECT_NAME)
	rm -rf releases/$(VERSION)/$(PROJECT_NAME)/* 2>/dev/null || true

# Copy deployment docker-compose.yml to release folder
	cp docker-compose.release.yml releases/$(VERSION)/$(PROJECT_NAME)/docker-compose.yml
	cp README.md releases/$(VERSION)/$(PROJECT_NAME)/README.md

# Update docker-compose.yml to use versioned images
	sed -i 's/deferred-diffusion-api:latest/$(REPO_USERNAME)\/$(REPO):api-$(VERSION)/g' releases/$(VERSION)/$(PROJECT_NAME)/docker-compose.yml
	sed -i 's/deferred-diffusion-workers:latest/$(REPO_USERNAME)\/$(REPO):worker-$(VERSION)/g' releases/$(VERSION)/$(PROJECT_NAME)/docker-compose.yml

# Copy directories with exclusions
	cp -r clients releases/$(VERSION)/$(PROJECT_NAME)/

# Create release archive
	cd releases/$(VERSION) && tar -czf $(PROJECT_NAME)-$(VERSION).tar.gz $(PROJECT_NAME)
	@echo Release files created in releases/$(VERSION)/$(PROJECT_NAME)
	@echo Archive created: releases/$(VERSION)/$(PROJECT_NAME)-$(VERSION).tar.gz