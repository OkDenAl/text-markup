package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"

	"github.com/OkDenAl/text-markup-gateway/internal/handler/responses"
	"github.com/OkDenAl/text-markup-gateway/internal/repo/ml-markup/httpl"
)

// @BasePath /api/v1

// getMarkupFromText godoc
// @Summary get markup from file
// @Schemes
// @Description get markup from file
// @Tags Markup
// @Accept multipart/form-data
// @Produce json
// @Param file formData file true "File to upload"
// @Success 200 {object} domain.TextEntities
// @Failure 400 {object} responses.HTTPError
// @Failure 500 {object} responses.HTTPError
// @Router /markup-file [post]
func getMarkupFromFile(markup iMLMarkup) gin.HandlerFunc {
	return func(c *gin.Context) {
		fileHeader, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, responses.Error(errors.Wrap(err, "failed to get request file")))
			_ = c.AbortWithError(http.StatusBadRequest, err)
			return
		}

		entities, err := markup.GetEntitiesFromFile(c, fileHeader)
		if err != nil {
			switch {
			case errors.Is(err, httpl.ErrInvalidData):
				c.JSON(http.StatusNoContent, responses.Error(err))
				_ = c.AbortWithError(http.StatusNoContent, err)
			case errors.Is(err, httpl.ErrInvalidFileExtension):
				c.JSON(http.StatusBadRequest, responses.Error(err))
				_ = c.AbortWithError(http.StatusBadRequest, err)
			default:
				c.JSON(http.StatusInternalServerError, responses.Error(err))
				_ = c.AbortWithError(http.StatusInternalServerError, err)
			}

			return
		}

		c.JSON(http.StatusOK, entities)
	}
}
