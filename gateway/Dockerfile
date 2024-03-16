FROM golang:1.21

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -buildvcs=false -o segmentService ./cmd/app/

CMD ["./segmentService"]