package middleware

import (
	"bytes"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"io"
	"net/http"
	"strconv"
	"time"
)

func Logger() gin.HandlerFunc {
	return func(c *gin.Context) {
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