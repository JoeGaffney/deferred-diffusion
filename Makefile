.PHONY: all build down up generate-clients test-texts test-images

# Default target
all: generate-clients

# Docker management
down:
	docker-compose down

build: down
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
	cd it_tests && pytest texts -vs
	cd ..
	docker-compose exec workers pytest tests/texts

# Test images modules
test-images: generate-clients
	cd it_tests && pytest images -vs
	cd ..
	docker-compose exec workers pytest tests/images

