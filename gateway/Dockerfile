FROM golang:1.21 as builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -tags=jsoniter -o gateway ./cmd/text-markup/main.go

FROM scratch
WORKDIR /
COPY --from=builder /app/gateway /gateway

EXPOSE 8080

CMD ["/gateway"]