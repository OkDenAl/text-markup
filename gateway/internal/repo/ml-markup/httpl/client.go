package httpl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/OkDenAl/text-markup-gateway/internal/config"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
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
	if err != nil {
		return MLResponse{}, err
	}

	req, err := http.NewRequest(
		"GET", fmt.Sprintf("http://%s:%s/api/v1/prediction", c.cfg.Host, c.cfg.Port), bytes.NewBuffer(reqJSON),
	)
	if err != nil {
		return MLResponse{}, err
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return MLResponse{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return MLResponse{}, err
	}

	var result MLResponse
	if err = json.Unmarshal(body, &result); err != nil {
		return MLResponse{}, err
	}

	return result, nil
}
