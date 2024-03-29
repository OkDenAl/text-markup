package main

import (
	"github.com/OkDenAl/text-markup-gateway/internal/config"
	"github.com/OkDenAl/text-markup-gateway/internal/handler"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/middleware"
	"github.com/gin-gonic/gin"
	"net/http"
	"os"
)

func newHTTPServer(cfg config.Server, h handler.Handler) *http.Server {
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	api := engine.Group("text-markup/v1", middleware.LoggerMiddleware(), gin.Recovery())
	h.SetRouter(api)
	return &http.Server{Addr: cfg.Port, Handler: engine}
}

func setupConfig() (*config.Config, error) {
	const (
		configPathEnv     = "CONFIG_PATH"
		defaultConfigPath = "./config/local_config.yml"
	)

	configPath := os.Getenv(configPathEnv)
	if configPath == "" {
		configPath = defaultConfigPath
	}

	return config.New(configPath)
}
