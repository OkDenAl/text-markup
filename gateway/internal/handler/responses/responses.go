package responses

import "github.com/gin-gonic/gin"

func Error(err error) *gin.H {
	return &gin.H{
		"data":  nil,
		"error": err.Error(),
	}
}

func Data(data any) *gin.H {
	return &gin.H{
		"data":  data,
		"error": nil,
	}
}
