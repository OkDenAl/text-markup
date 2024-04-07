package main

import (
	"net/http"
	"os"

	"github.com/ds248a/closer"
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"

	"github.com/OkDenAl/text-markup-gateway/internal/config"
	"github.com/OkDenAl/text-markup-gateway/internal/handler"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/middleware"
	"github.com/OkDenAl/text-markup-gateway/pkg/logger"
)

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

func setupLogger(cfg *config.Config) {
	logger.SetupLogLevel(cfg.LogLevel)
	logger.SetupWriter()
}

func initAndStartHTTPServer(cfg config.ServerConfig, h handler.Handler) <-chan error {
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	if cfg.SwaggerEnabled != nil && *cfg.SwaggerEnabled {
		engine.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	}

	api := engine.Group("api/v1")
	api.Use(
		gin.Recovery(),
		middleware.CORS(),
		middleware.Logger(),
	)
	h.SetRouter(api)

	s := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      engine,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	closer.Add(func() {
		if err := s.Close(); err != nil {
			log := logger.New()
			log.Error().Stack().Err(err).Msg("failed to stop http server")
		}
	})

	errCh := make(chan error)
	go func() {
		if err := s.ListenAndServe(); err != nil {
			errCh <- errors.WithStack(err)
			close(errCh)
		}
	}()

	return errCh
}

func printLocalURLS(port string) {
	log := logger.New()

	log.Debug().Msgf("HTTP: http://localhost:%s", port)
	log.Debug().Msgf("SWAGGER: http://localhost:%s/swagger/index.html", port)
}
