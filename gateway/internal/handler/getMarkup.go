package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/responses"
	"github.com/OkDenAl/text-markup-gateway/pkg/logger"
)

// @BasePath /api/v1

// getMarkup godoc
// @Summary get markup from text (string)
// @Schemes
// @Description get markup from text (string)
// @Tags Markup
// @Produce json
// @Param   text body model.TextMarkupRequest  true  "Text JSON"
// @Success 200 {object} domain.TextEntities
// @Failure 500 {object} responses.HTTPError
// @Router /markup [post]
func getMarkup(markup iMLMarkup) gin.HandlerFunc {
	return func(c *gin.Context) {
		log := logger.New()
		req := model.TextMarkupRequest{}
		if err := c.BindJSON(&req); err != nil {
			log.Error().Stack().Err(err).Msg("failed to get request data")
			c.JSON(http.StatusBadRequest, responses.Error(err))
			return
		}

		link, err := markup.GetEntitiesFromText(c, req.Text)
		if err != nil {
			log.Error().Stack().Err(err).Msg("failed to get entities from text")
			c.JSON(http.StatusInternalServerError, responses.Error(err))
			return
		}

		c.JSON(http.StatusOK, link)
	}
}
