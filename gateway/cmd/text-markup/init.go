package main

import (
	"github.com/gin-gonic/gin"
	"net/http"
)

func newHTTPServer(port string) *http.Server {
	gin.SetMode(gin.ReleaseMode)
	handler := gin.New()
	api := handler.Group("text-markup/v1", handler.LoggerMiddleware(log), gin.Recovery())
	{
		segmentPort.SetRouter(api, services.SegmentService)
		userSegmentPort.SetRouter(api, services.UserSegmentService)
		operationPort.SetRouter(api, services.OperationService)
	}
	return &http.Server{Addr: port, Handler: handler}
}
