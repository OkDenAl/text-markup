package handler

import (
	"context"
	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"github.com/pkg/errors"
)

var ErrValidationFailed = errors.New("validation failed")

//go:generate minimock -g -s .go -i iUsecase -o ../../mocks/handler
type iMLMarkup interface {
	GetEntitiesFromText(ctx context.Context, text string) (domain.TextEntities, error)
}

type Handler struct {
	mlMarkup iMLMarkup
}

func NewHandler(mlMarkup iMLMarkup) (Handler, error) {
	if mlMarkup == nil {
		return Handler{}, ErrValidationFailed
	}

	return Handler{mlMarkup}, nil
}
