package handler

import (
	"context"
	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"github.com/gin-gonic/gin"
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

func New(mlMarkup iMLMarkup) (Handler, error) {
	if mlMarkup == nil {
		return Handler{}, ErrValidationFailed
	}

	return Handler{mlMarkup}, nil
}

func (h Handler) SetRouter(api *gin.RouterGroup) {
	api.POST("/markup", getMarkup(h.mlMarkup))
}
