package main

import (
	"os"
	"os/signal"
	"syscall"

	_ "github.com/OkDenAl/text-markup-gateway/docs"
	"github.com/ds248a/closer"

	"github.com/OkDenAl/text-markup-gateway/internal/handler"
	"github.com/OkDenAl/text-markup-gateway/internal/repo/ml-markup/httpl"
	"github.com/OkDenAl/text-markup-gateway/pkg/logger"
)

// @title           Text Markup Service
// @version         1.0
// @description     Text markup - it is the service for getting markup from text.

// @contact.name   text-markup
// @contact.url    https://github.com/OkDenAl/text-markup

// @BasePath  /api/v1
func main() {
	defer func() {
		if recover() != nil {
			os.Exit(1)
		}
	}()

	cfg, err := setupConfig()
	if err != nil {
		log := logger.New()
		log.Panic().Stack().Err(err).Msg("failed to setup cfg")
	}

	setupLogger(cfg)
	log := logger.New()

	mlClient := httpl.NewClient(cfg.MLClient)

	mlMarkupRepo, err := httpl.NewMLMarkupRepo(mlClient)
	if err != nil {
		log.Panic().Stack().Err(err).Msg("failed to setup mlMarkupRepo")
	}

	h, err := handler.New(mlMarkupRepo)
	if err != nil {
		log.Panic().Stack().Err(err).Msg("failed to setup handler")
	}

	errCh := initAndStartHTTPServer(cfg.HTTP, h)
	printLocalURLS(cfg.HTTP.Port)

	gracefulShutdown(errCh)
}

func gracefulShutdown(errCh <-chan error) {
	log := logger.New()
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)
	select {
	case s := <-signals:
		closer.Close(s)
		log.Error().Stack().Msgf("os signal detected - %s", s.String())
	case err := <-errCh:
		closer.Close(syscall.SIGTERM)
		log.Error().Stack().Err(err).Msgf("http server error detected")
	}
}
