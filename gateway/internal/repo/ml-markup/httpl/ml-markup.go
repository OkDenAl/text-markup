package httpl

import (
	"context"

	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
)

type MLMarkupRepo struct {
	client iMLClient
}

func NewMLMarkupRepo(client iMLClient) (MLMarkupRepo, error) {
	return MLMarkupRepo{client: client}, nil
}

func (r MLMarkupRepo) GetEntitiesFromText(ctx context.Context, text string) (te domain.TextEntities, err error) {
	var resp MLResponse
	resp, err = r.client.GetPrediction(ctx, model.NewTextMarkupRequest(text))
	if err != nil {
		return domain.TextEntities{}, err
	}
	for i, label := range resp.Labels {
		if label != "O" {
			te.Labels = append(te.Labels, resp.Labels[i])
			te.Tags = append(te.Tags, resp.Tokens[i])
		}
	}

	return te, nil
}
