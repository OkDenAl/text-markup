FROM golang:1.21 as builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o gateway ./cmd/text-markup/


EXPOSE 8000

CMD ["./gateway"]