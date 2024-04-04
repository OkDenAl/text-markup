.PHONY: start stop cover lint build prod

start:
	docker-compose up -d --build

stop:
	docker-compose down

cover:
	go test -v -coverpkg=./... -coverprofile report.out -covermode=atomic ./...
	grep -v -E -- 'mocks|config|cmd|logging'  report.out > report1.out
	go tool cover -func=report1.out

lint:
	golangci-lint run ./...

prod:
	@mv ./docker-compose.yml ./docker-compose-dev.yml
	@cp ./prod/docker-compose.yml ./docker-compose.yml
	@docker compose up --build -d
	@rm ./docker-compose.yml
	@mv ./docker-compose-dev.yml ./docker-compose.yml

