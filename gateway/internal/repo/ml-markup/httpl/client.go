package httpl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/OkDenAl/text-markup-gateway/internal/config"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
	"github.com/rs/zerolog/log"
	"io"
	"net/http"
)

type iMLClient interface {
	GetPrediction(request model.TextMarkupRequest) (MLResponse, error)
}

type MlClient struct {
	cfg    config.ClientConfig
	client http.Client
}

func NewClient(cfg config.ClientConfig) MlClient {
	return MlClient{client: http.Client{}, cfg: cfg}
}

func (c MlClient) GetPrediction(reqData model.TextMarkupRequest) (MLResponse, error) {
	reqJSON, err := json.Marshal(reqData)

	req, err := http.NewRequest(
		"GET", fmt.Sprintf("http://%s:%s/get_prediction", c.cfg.Host, c.cfg.Port), bytes.NewBuffer(reqJSON),
	)
	if err != nil {
		return MLResponse{}, err
	}

	resp, err := c.client.Do(req)
	if err != nil {
		log.Error().Stack().Err(err).Msg("error")
		return MLResponse{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)

	var result MLResponse
	err = json.Unmarshal([]byte(body), &result)
	if err != nil {
		return MLResponse{}, err
	}

	return result, nil
}
