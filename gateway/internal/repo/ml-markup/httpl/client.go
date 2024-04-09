package httpl

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/benbjohnson/clock"
	"github.com/cenkalti/backoff/v3"
	"github.com/mercari/go-circuitbreaker"
	"github.com/pkg/errors"

	"github.com/OkDenAl/text-markup-gateway/internal/config"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
)

type iMLClient interface {
	GetPrediction(ctx context.Context, request model.TextMarkupRequest) (MLResponse, error)
}

type MlClient struct {
	cfg    config.ClientConfig
	client http.Client
	cb     *circuitbreaker.CircuitBreaker
}

func NewClient(cfg config.ClientConfig) MlClient {
	cb := circuitbreaker.New(
		circuitbreaker.WithClock(clock.New()),
		circuitbreaker.WithFailOnContextCancel(true),
		circuitbreaker.WithFailOnContextDeadline(true),
		circuitbreaker.WithHalfOpenMaxSuccesses(cfg.CircuitBreaker.HalfOpenMaxSuccesses),
		circuitbreaker.WithOpenTimeoutBackOff(backoff.NewExponentialBackOff()),
		circuitbreaker.WithCounterResetInterval(cfg.CircuitBreaker.CounterResetInterval),
		circuitbreaker.WithTripFunc(
			circuitbreaker.NewTripFuncFailureRate(cfg.CircuitBreaker.MinThreshold, cfg.CircuitBreaker.FailureRate),
		),
	)

	return MlClient{client: http.Client{}, cfg: cfg, cb: cb}
}

func (c MlClient) GetPrediction(ctx context.Context, reqData model.TextMarkupRequest) (resp MLResponse, err error) {
	if !c.cb.Ready() {
		return MLResponse{}, circuitbreaker.ErrOpen
	}
	defer func() { err = c.cb.Done(ctx, err) }()

	var reqJSON []byte
	reqJSON, err = json.Marshal(reqData)
	if err != nil {
		return MLResponse{}, errors.Wrap(err, "failed to marshal req data")
	}

	var req *http.Request
	req, err = http.NewRequest(
		"GET", fmt.Sprintf("http://%s:%s/api/v1/prediction", c.cfg.Host, c.cfg.Port), bytes.NewBuffer(reqJSON),
	)
	if err != nil {
		return MLResponse{}, errors.Wrapf(err, "failed to create http request to %s", req.URL.String())
	}

	var clientResp *http.Response
	clientResp, err = c.client.Do(req)
	if err != nil {
		return MLResponse{}, errors.Wrapf(err, "failed to send http request to %s", req.URL.String())
	}
	defer clientResp.Body.Close()

	var body []byte
	body, err = io.ReadAll(clientResp.Body)
	if err != nil {
		return MLResponse{}, errors.Wrap(err, "failed to read response body")
	}

	var result MLResponse
	if err = json.Unmarshal(body, &result); err != nil {
		return MLResponse{}, errors.Wrap(err, "failed unmarshal response data")
	}

	return result, nil
}
