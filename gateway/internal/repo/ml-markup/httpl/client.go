package httpl

import (
	"bytes"
	"encoding/json"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
	"io"
	"log"
	"net/http"
)

type iMLClient interface {
	GetPrediction(request model.TextMarkupRequest) (MLResponse, error)
}

type MlClient struct {
	client http.Client
}

func NewClient() MlClient {
	return MlClient{client: http.Client{}}
}

func (c MlClient) GetPrediction(reqData model.TextMarkupRequest) (MLResponse, error) {
	reqJSON, err := json.Marshal(reqData)
	log.Println("here")

	req, err := http.NewRequest(
		"GET", "http://127.0.0.1:8091/get_prediction", bytes.NewBuffer(reqJSON),
	)

	resp, err := c.client.Do(req)
	if err != nil {
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
