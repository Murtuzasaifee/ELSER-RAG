.PHONY: help build up up-build down restart logs logs-app logs-es ps shell clean nuke

help:
	@echo "ELSER-RAG — Docker commands"
	@echo ""
	@echo "  make build       Build images"
	@echo "  make up          Start all services (detached)"
	@echo "  make up-build    Build then start all services (detached)"
	@echo "  make down        Stop and remove containers"
	@echo "  make restart     Restart app container only"
	@echo "  make logs        Tail logs for all services"
	@echo "  make logs-app    Tail app logs"
	@echo "  make logs-es     Tail Elasticsearch logs"
	@echo "  make ps          Show running containers"
	@echo "  make shell       Open bash shell in app container"
	@echo "  make clean       Remove containers, local images, volumes"
	@echo "  make nuke        Remove containers, ALL images, volumes"

build:
	docker compose build

up:
	docker compose up -d

up-build:
	docker compose up -d --build

down:
	docker compose down

restart:
	docker compose restart app

logs:
	docker compose logs -f

logs-app:
	docker compose logs -f app

logs-es:
	docker compose logs -f elasticsearch

ps:
	docker compose ps

shell:
	docker compose exec app bash

clean:
	docker compose down --rmi local --volumes --remove-orphans

nuke:
	docker compose down --rmi all --volumes --remove-orphans
