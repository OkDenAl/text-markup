package httpl

import (
	"context"
	"github.com/pkg/errors"

	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
)

var (
	ErrInvalidData = errors.New("invalid data")
)

type MLMarkupRepo struct {
	client iMLClient
}

func NewMLMarkupRepo(client iMLClient) (MLMarkupRepo, error) {
	return MLMarkupRepo{client: client}, nil
}

func (r MLMarkupRepo) GetTokensFromText(ctx context.Context, text string) (domain.Tokens, error) {
	tokens, err := r.client.GetTokens(ctx, model.NewTextMarkupRequest(text))
	if err != nil {
		return domain.Tokens{}, err
	}

	if len(tokens.Tags) == 0 && len(tokens.Labels) == 0 {
		return domain.Tokens{}, errors.Wrap(ErrInvalidData, "failed to get tokens from text")
	}

	return tokens, nil
}

func (r MLMarkupRepo) GetClassFromText(ctx context.Context, text string) (domain.Class, error) {
	class, err := r.client.GetClass(ctx, model.NewTextMarkupRequest(text))
	if err != nil {
		return domain.Class{}, err
	}

	if class.Class == "" {
		return domain.Class{}, errors.Wrap(ErrInvalidData, "failed to get class from text")
	}

	return class, nil
}

func (r MLMarkupRepo) GetKeywordsFromText(ctx context.Context, text string) (domain.Keywords, error) {
	const topN = 5
	keywordFromML, err := r.client.GetKeywords(ctx, model.NewTextKeywordsRequest(text, topN))
	if err != nil {
		return domain.Keywords{}, err
	}

	if len(keywordFromML.Keywords) == 0 {
		return domain.Keywords{}, errors.Wrap(ErrInvalidData, "failed to get keywords from text")
	}

	return domain.NewKeywords(keywordFromML.Keywords), nil
}
