.PHONY:  all down copy-schemas build up generate-clients test-worker test-it-tests create-release mypy-check

VERSION ?= latest
PROJECT_NAME ?= deferred-diffusion
REPO ?= deferred-diffusion
REPO_USERNAME ?= joegaffney
TEST_PATH ?= images/models


# Default target
all: generate-clients

down:
	docker compose down

# NOTE atm we align by copying schemas from the api to the workers
copy-schemas:
ifeq ($(OS),Windows_NT)
	copy api\images\schemas.py workers\images\schemas.py
	copy api\texts\schemas.py workers\texts\schemas.py
	copy api\videos\schemas.py workers\videos\schemas.py
	copy api\common\schemas.py workers\common\schemas.py
	copy api\workflows\schemas.py workers\workflows\schemas.py
else
	cp api/images/schemas.py workers/images/schemas.py
	cp api/texts/schemas.py workers/texts/schemas.py
	cp api/videos/schemas.py workers/videos/schemas.py
	cp api/common/schemas.py workers/common/schemas.py
	cp api/workflows/schemas.py workers/workflows/schemas.py
endif

build: down copy-schemas
	docker compose build

up: build
	docker compose up -d

up-it-tests: down copy-schemas
	docker compose -f docker-compose.it-tests.yml build
	docker compose -f docker-compose.it-tests.yml up -d

up-comfy:
	docker compose -f docker-compose.comfy.yml down
	docker compose -f docker-compose.comfy.yml build
	docker compose -f docker-compose.comfy.yml up -d

up-latest-release:
	docker compose down
	docker compose pull
	docker compose up -d --no-build

# Generate OpenAPI spec file
generate-openapi-spec: up
	curl -o clients/openapi.json http://127.0.0.1:5000/openapi.json || powershell -Command "Invoke-WebRequest -Uri 'http://127.0.0.1:5000/openapi.json' -OutFile 'clients/openapi.json'"
	@echo OpenAPI spec saved to clients/openapi.json

# API Client generation
generate-clients-raw: 
	openapi-python-client generate --path clients/openapi.json --output-path clients/houdini/python/generated --overwrite
	openapi-python-client generate --path clients/openapi.json --output-path clients/nuke/python/generated --overwrite
	openapi-python-client generate --path clients/openapi.json --output-path clients/it_tests/generated --overwrite

generate-clients: generate-openapi-spec generate-clients-raw
	@echo "API clients generated successfully."

# Mypy type checks
mypy-api:
	docker compose exec api mypy .

mypy-worker:
	docker compose exec gpu-workers mypy .

mypy-cpu-workers:
	docker compose exec cpu-workers mypy .

# Example test commands:
# make test-worker TEST_PATH=images
# make test-worker TEST_PATH=images/local/test_flux_1.py
# make test-worker TEST_PATH=images/external/test_flux_1_pro.py
# make test-worker TEST_PATH=texts
# make test-worker TEST_PATH=videos
# make test-worker TEST_PATH=videos/local/test_wan_2.py
# make test-worker TEST_PATH=videos/external/test_runway_gen4.py
test-worker: up
	docker compose exec gpu-workers pytest tests/$(TEST_PATH) -vs

test-worker-basic: up
	docker compose exec gpu-workers pytest -m "basic" -vs

test-worker-workflows: up-comfy up 
	docker compose exec gpu-workers pytest tests/workflows -vs

# make it-tests TEST_PATH=test_image_local.py
it-tests:
	cd clients/it_tests && pytest tests/$(TEST_PATH) -vs
	cd ../..

it-tests-basic:
	cd clients/it_tests && pytest -m "basic" -vs
	cd ../..

# Create release package
create-client-release: generate-clients-raw
	python scripts/package_release.py $(VERSION) $(PROJECT_NAME)




