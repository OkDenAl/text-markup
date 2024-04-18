package handler

import (
	"context"
	"errors"
	"github.com/gin-gonic/gin"

	"github.com/OkDenAl/text-markup-gateway/internal/domain"
)

var ErrValidationFailed = errors.New("validation failed")

//go:generate minimock -g -s .go -i iUsecase -o ../../mocks/handler
type iMLMarkup interface {
	GetTokensFromText(ctx context.Context, text string) (domain.Tokens, error)
	GetClassFromText(ctx context.Context, text string) (domain.Class, error)
}

type Handler struct {
	mlMarkup iMLMarkup
}

func New(mlMarkup iMLMarkup) (Handler, error) {
	if mlMarkup == nil {
		return Handler{}, ErrValidationFailed
	}

	return Handler{mlMarkup}, nil
}

func (h Handler) SetRouter(api *gin.RouterGroup) {
	api.POST("/markup", getMarkup(h.mlMarkup))
	api.POST("/markup-file", getMarkupFromFile(h.mlMarkup))
}
