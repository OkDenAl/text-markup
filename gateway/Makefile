.PHONY: start stop cover lint

start:
	docker compose up -d --build

stop:
	docker compose down

cover:
	go test -v -coverpkg=./... -coverprofile report.out -covermode=atomic ./...
	grep -v -E -- 'mocks|config|cmd|logging'  report.out > report1.out
	go tool cover -func=report1.out

lint:
	golangci-lint run

test:
	go test -v ./...

mockgen:
	go generate ./...
