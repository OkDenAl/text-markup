package main

import (
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	"github.com/OkDenAl/text-markup-gateway/internal/config"
	"github.com/OkDenAl/text-markup-gateway/internal/handler"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/middleware"
)

func newHTTPServer(cfg config.ServerConfig, h handler.Handler) *http.Server {
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	api := engine.Group("api/v1", middleware.Logger(), middleware.CORS(), gin.Recovery())
	h.SetRouter(api)
	return &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      engine,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}
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
