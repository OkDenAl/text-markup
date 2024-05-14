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
	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
)

type iMLClient interface {
	GetTokens(ctx context.Context, request model.TextMarkupRequest) (domain.Tokens, error)
	GetClass(ctx context.Context, request model.TextMarkupRequest) (domain.Class, error)
	GetKeywords(ctx context.Context, reqData model.TextKeywordsRequest) (keywordsResp TextKeywordsResponse, err error)
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

func (c MlClient) GetTokens(ctx context.Context, reqData model.TextMarkupRequest) (tokens domain.Tokens, err error) {
	if !c.cb.Ready() {
		return domain.Tokens{}, circuitbreaker.ErrOpen
	}
	defer func() { err = c.cb.Done(ctx, err) }()

	var reqJSON []byte
	reqJSON, err = json.Marshal(reqData)
	if err != nil {
		return domain.Tokens{}, errors.Wrap(err, "failed to marshal req data")
	}

	var req *http.Request
	req, err = http.NewRequest(
		"GET", fmt.Sprintf("http://%s:%s/api/v1/tokens", c.cfg.Host, c.cfg.Port),
		bytes.NewBuffer(reqJSON),
	)
	if err != nil {
		return domain.Tokens{}, errors.Wrapf(err, "failed to create http request to %s", req.URL.String())
	}

	var clientResp *http.Response
	clientResp, err = c.client.Do(req)
	if err != nil {
		return domain.Tokens{}, errors.Wrapf(err, "failed to send http request to %s", req.URL.String())
	}
	defer clientResp.Body.Close()

	var body []byte
	body, err = io.ReadAll(clientResp.Body)
	if err != nil {
		return domain.Tokens{}, errors.Wrap(err, "failed to read response body")
	}

	if err = json.Unmarshal(body, &tokens); err != nil {
		return domain.Tokens{}, errors.Wrap(err, "failed to unmarshal response data")
	}

	return tokens, nil
}

func (c MlClient) GetClass(ctx context.Context, reqData model.TextMarkupRequest) (class domain.Class, err error) {
	if !c.cb.Ready() {
		return domain.Class{}, circuitbreaker.ErrOpen
	}
	defer func() { err = c.cb.Done(ctx, err) }()

	var reqJSON []byte
	reqJSON, err = json.Marshal(reqData)
	if err != nil {
		return domain.Class{}, errors.Wrap(err, "failed to marshal req data")
	}

	var req *http.Request
	req, err = http.NewRequest(
		"GET", fmt.Sprintf("http://%s:%s/api/v1/class", c.cfg.Host, c.cfg.Port), bytes.NewBuffer(reqJSON),
	)
	if err != nil {
		return domain.Class{}, errors.Wrapf(err, "failed to create http request to %s", req.URL.String())
	}

	var clientResp *http.Response
	clientResp, err = c.client.Do(req)
	if err != nil {
		return domain.Class{}, errors.Wrapf(err, "failed to send http request to %s", req.URL.String())
	}
	defer clientResp.Body.Close()

	var body []byte
	body, err = io.ReadAll(clientResp.Body)
	if err != nil {
		return domain.Class{}, errors.Wrap(err, "failed to read response body")
	}

	if err = json.Unmarshal(body, &class); err != nil {
		return domain.Class{}, errors.Wrap(err, "failed to unmarshal response data")
	}

	return class, nil
}

func (c MlClient) GetKeywords(
	ctx context.Context,
	reqData model.TextKeywordsRequest,
) (keywordsResp TextKeywordsResponse, err error) {
	if !c.cb.Ready() {
		return TextKeywordsResponse{}, circuitbreaker.ErrOpen
	}
	defer func() { err = c.cb.Done(ctx, err) }()

	var reqJSON []byte
	reqJSON, err = json.Marshal(reqData)
	if err != nil {
		return TextKeywordsResponse{}, errors.Wrap(err, "failed to marshal req data")
	}

	var req *http.Request
	req, err = http.NewRequest(
		"GET", fmt.Sprintf("http://%s:%s/api/v1/keywords", c.cfg.Host, c.cfg.Port),
		bytes.NewBuffer(reqJSON),
	)
	if err != nil {
		return TextKeywordsResponse{}, errors.Wrapf(err, "failed to create http request to %s", req.URL.String())
	}

	var clientResp *http.Response
	clientResp, err = c.client.Do(req)
	if err != nil {
		return TextKeywordsResponse{}, errors.Wrapf(err, "failed to send http request to %s", req.URL.String())
	}
	defer clientResp.Body.Close()

	var body []byte
	body, err = io.ReadAll(clientResp.Body)
	if err != nil {
		return TextKeywordsResponse{}, errors.Wrap(err, "failed to read response body")
	}

	if err = json.Unmarshal(body, &keywordsResp); err != nil {
		return TextKeywordsResponse{}, errors.Wrap(err, "failed to unmarshal response data")
	}

	return keywordsResp, nil
}
