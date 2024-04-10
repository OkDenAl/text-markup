package handler

import (
	"github.com/OkDenAl/text-markup-gateway/internal/repo/ml-markup/httpl"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"

	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/responses"
)

// @BasePath /api/v1

// getMarkup godoc
// @Summary get markup from text
// @Schemes
// @Description get markup from text
// @Tags Markup
// @Produce json
// @Param   text body model.TextMarkupRequest  true  "Text JSON"
// @Success 200 {object} domain.TextEntities
// @Success 204 {object} responses.HTTPError
// @Failure 500 {object} responses.HTTPError
// @Router /markup [post]
func getMarkup(markup iMLMarkup) gin.HandlerFunc {
	return func(c *gin.Context) {
		req := model.TextMarkupRequest{}
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, responses.Error(errors.Wrap(err, "failed to get request data")))
			_ = c.AbortWithError(http.StatusBadRequest, err)
			return
		}

		entities, err := markup.GetEntitiesFromText(c, req.Text)
		if err != nil {
			switch {
			case errors.Is(err, httpl.ErrInvalidData):
				c.JSON(http.StatusNoContent, responses.Error(err))
				_ = c.AbortWithError(http.StatusNoContent, err)
			default:
				c.JSON(http.StatusInternalServerError, responses.Error(err))
				_ = c.AbortWithError(http.StatusInternalServerError, err)
			}
			return
		}

		c.JSON(http.StatusOK, entities)
	}
}
