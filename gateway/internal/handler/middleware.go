package handler

import (
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"time"
)

func LoggerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		t := time.Now()
		c.Next()
		latency := time.Since(t)
		status := c.Writer.Status()
		log.Print("Latency:", latency, "\tMethod:", c.Request.Method, "\tPath:",
			c.Request.URL.Path, "\tStatus:", status)
	}
}
