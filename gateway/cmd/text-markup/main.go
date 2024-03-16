package main

import (
	"context"
	"errors"
	"fmt"
	"github.com/OkDenAl/text-markup-gateway/internal/handler"
	"github.com/OkDenAl/text-markup-gateway/internal/repo/ml-markup/httpl"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	defer func() {
		if recover() != nil {
			os.Exit(1)
		}
	}()

	cfg, err := setupConfig()
	if err != nil {
		fmt.Println(err)
		log.Fatal().Msg("cant setup cfg")
	}

	mlClient := httpl.NewClient()

	mlMarkupRepo, err := httpl.NewMLMarkupRepo(mlClient)
	if err != nil {
		fmt.Println(err)
		log.Fatal().Msg("cant setup mlMarkupRepo")
	}

	h, err := handler.New(mlMarkupRepo)
	if err != nil {
		log.Fatal().Msg("cant setup handler")
	}

	server := newHTTPServer(cfg.Server, h)
	g, ctx := errgroup.WithContext(context.Background())
	gracefulShutdown(ctx, g)

	g.Go(func() error {
		log.Print("starting httpl server on port: %s\n", server.Addr)
		defer log.Print("closing httpl server on port: %s\n", server.Addr)

		errCh := make(chan error)

		defer func() {
			shCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			if err := server.Shutdown(shCtx); err != nil {
				log.Print("can't close httpl server listening on %s: %s", server.Addr, err.Error())
			}

			close(errCh)
		}()

		go func() {
			if err = server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
				fmt.Println(err)
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
		log.Print("gracefully shutting down the server: %s\n", err.Error())
	}
}

func gracefulShutdown(ctx context.Context, g *errgroup.Group) {
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM, syscall.SIGQUIT)
	g.Go(func() error {
		select {
		case s := <-signals:
			return fmt.Errorf("captured signal %s\n", s)
		case <-ctx.Done():
			return nil
		}
	})
}
