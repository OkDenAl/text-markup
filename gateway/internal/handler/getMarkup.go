package handler

import (
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/responses"
	"github.com/gin-gonic/gin"
	"log"
	"net/http"
)

func getMarkup(markup iMLMarkup) gin.HandlerFunc {
	return func(c *gin.Context) {
		req := model.TextMarkupRequest{}
		log.Println(req)
		err := c.BindJSON(&req)
		if err != nil {
			c.JSON(http.StatusBadRequest, responses.Error(err))
			return
		}
		link, err := markup.GetEntitiesFromText(c, req.Text)
		if err != nil {
			switch err {
			default:
				c.JSON(http.StatusInternalServerError, responses.Error(err))
				return
			}
		}
		c.JSON(http.StatusOK, responses.Data(link))
	}
}
