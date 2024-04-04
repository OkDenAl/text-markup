.PHONY: start stop cover lint build prod-stop prod-start

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

prod-stop:
	@mv ./docker-compose.yml ./docker-compose-dev.yml
	@cp ./prod/docker-compose.yml ./docker-compose.yml
	@make stop
	@rm ./docker-compose.yml
	@mv ./docker-compose-dev.yml ./docker-compose.yml

prod-start:
	@mv ./docker-compose.yml ./docker-compose-dev.yml
	@cp ./prod/docker-compose.yml ./docker-compose.yml
	@make start
	@rm ./docker-compose.yml
	@mv ./docker-compose-dev.yml ./docker-compose.yml

