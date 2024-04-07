package middleware

import (
	"bytes"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/OkDenAl/text-markup-gateway/pkg/logger"
)

func Logger() gin.HandlerFunc {
	return func(c *gin.Context) {
		log := logger.New()
		var reqBody []byte
		c.Request.Body, reqBody = readBody(c.Request.Body, c.Request.Header)

		t := time.Now()
		c.Next()
		latency := time.Since(t).String()

		status := c.Writer.Status()
		log.Info().
			Str("latency", latency).
			Str("method:", c.Request.Method).
			Str("path", c.Request.URL.Path).
			Str("status", strconv.Itoa(status)).
			RawJSON("body", reqBody).
			Msg("request done")
	}
}

func readBody(body io.ReadCloser, hdrs http.Header) (newBody io.ReadCloser, content []byte) {
	if !contentTypeIsJSON(hdrs) {
		return body, nil
	}

	bodyBytes, err := io.ReadAll(body)
	if err != nil {
		return body, nil
	}

	if err := body.Close(); err != nil {
		return body, nil
	}

	return io.NopCloser(bytes.NewBuffer(bodyBytes)), bodyBytes
}

func contentTypeIsJSON(hdrs http.Header) bool {
	if hdrs == nil {
		return false
	}

	return hdrs.Get("Content-Type") == "application/json"
}
