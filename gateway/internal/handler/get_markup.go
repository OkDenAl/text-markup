package handler

import (
	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
	"golang.org/x/sync/errgroup"

	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/responses"
	"github.com/OkDenAl/text-markup-gateway/internal/repo/ml-markup/httpl"
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
		err := c.BindJSON(&req)
		if err != nil {
			c.JSON(http.StatusBadRequest, responses.Error(errors.Wrap(err, "failed to get request data")))
			_ = c.AbortWithError(http.StatusBadRequest, err)
			return
		}

		var (
			eg     errgroup.Group
			tokens domain.Tokens
			class  domain.Class
		)

		eg.Go(func() error {
			tokens, err = markup.GetTokensFromText(c, req.Text)
			return err
		})

		eg.Go(func() error {
			class, err = markup.GetClassFromText(c, req.Text)
			return err
		})

		if err = eg.Wait(); err != nil {
			switch {
			case errors.Is(err, httpl.ErrInvalidData):
				c.JSON(http.StatusNoContent, nil)
				_ = c.AbortWithError(http.StatusNoContent, err)
			default:
				c.JSON(http.StatusInternalServerError, responses.Error(err))
				_ = c.AbortWithError(http.StatusInternalServerError, err)
			}
			return
		}

		c.JSON(http.StatusOK, domain.NewTextEntities(class.Class, tokens.Labels, tokens.Tags))
	}
}
