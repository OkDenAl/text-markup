package httpl

import (
	"context"
	"errors"
	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
)

var ErrValidationFailed = errors.New("validation failed")

type MLMarkupRepo struct {
	client iMLClient
}

func NewMLMarkupRepo(client iMLClient) (MLMarkupRepo, error) {
	return MLMarkupRepo{client: client}, nil
}

func (r MLMarkupRepo) GetEntitiesFromText(ctx context.Context, text string) (domain.TextEntities, error) {
	resp, err := r.client.GetPrediction(model.NewTextMarkupRequest(text))
	if err != nil {
		return domain.TextEntities{}, err
	}
	var te domain.TextEntities
	for i, label := range resp.Labels {
		if label != "O" {
			te.Tags = append(te.Tags, resp.Tokens[i])
		}
	}

	return te, nil
}
