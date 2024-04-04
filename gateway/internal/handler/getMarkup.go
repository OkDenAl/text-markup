package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/responses"
)

func getMarkup(markup iMLMarkup) gin.HandlerFunc {
	return func(c *gin.Context) {
		req := model.TextMarkupRequest{}
		err := c.BindJSON(&req)
		if err != nil {
			c.JSON(http.StatusBadRequest, responses.Error(err))
			return
		}
		link, err := markup.GetEntitiesFromText(c, req.Text)
		if err != nil {
			c.JSON(http.StatusInternalServerError, responses.Error(err))
			return
		}
		c.JSON(http.StatusOK, responses.Data(link))
	}
}
