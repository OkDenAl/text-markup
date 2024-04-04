package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"

	"github.com/OkDenAl/text-markup-gateway/internal/handler"
	"github.com/OkDenAl/text-markup-gateway/internal/repo/ml-markup/httpl"
)

func main() {
	defer func() {
		if recover() != nil {
			os.Exit(1)
		}
	}()

	cfg, err := setupConfig()
	if err != nil {
		log.Panic().Stack().Err(err).Msg("failed to setup cfg")
	}

	mlClient := httpl.NewClient(cfg.MLClient)

	mlMarkupRepo, err := httpl.NewMLMarkupRepo(mlClient)
	if err != nil {
		log.Panic().Stack().Err(err).Msg("failed to setup mlMarkupRepo")
	}

	h, err := handler.New(mlMarkupRepo)
	if err != nil {
		log.Panic().Stack().Err(err).Msg("failed to setup handler")
	}

	server := newHTTPServer(cfg.HTTP, h)
	g, ctx := errgroup.WithContext(context.Background())
	gracefulShutdown(ctx, g)

	g.Go(func() error {
		log.Info().Msgf("starting httpl server on port: %s\n", server.Addr)
		defer log.Info().Msgf("closing httpl server on port: %s\n", server.Addr)

		errCh := make(chan error)

		defer func() {
			const cancelTime = 30 * time.Second
			shCtx, cancel := context.WithTimeout(context.Background(), cancelTime)
			defer cancel()

			if err = server.Shutdown(shCtx); err != nil {
				log.Error().Stack().Err(err).Msgf("can't close http server listening on %s", server.Addr)
			}

			close(errCh)
		}()

		go func() {
			if err = server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
				errCh <- err
			}
		}()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err = <-errCh:
			return fmt.Errorf("httpl server can't listen and serve requests: %w", err)
		}
	})

	if err = g.Wait(); err != nil {
		log.Panic().Stack().Err(err).Msg("gracefully shutting down the server")
	}
}

func gracefulShutdown(ctx context.Context, g *errgroup.Group) {
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM, syscall.SIGQUIT)
	g.Go(func() error {
		select {
		case s := <-signals:
			return fmt.Errorf("captured signal %s", s.String())
		case <-ctx.Done():
			return nil
		}
	})
}
