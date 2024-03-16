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
)

func main() {
	server := NewServe

	g, ctx := errgroup.WithContext(context.Background())
	gracefulShutdown(ctx, g)

	g.Go(func() error {
		log.Infof("starting http server on port: %s\n", server.Addr)
		defer log.Infof("closing http server on port: %s\n", server.Addr)

		errCh := make(chan error)

		defer func() {
			shCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			if err := server.Shutdown(shCtx); err != nil {
				log.Infof("can't close http server listening on %s: %s", server.Addr, err.Error())
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
			return fmt.Errorf("http server can't listen and serve requests: %w", err)
		}
	})

	if err = g.Wait(); err != nil {
		log.Infof("gracefully shutting down the server: %s\n", err.Error())
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
