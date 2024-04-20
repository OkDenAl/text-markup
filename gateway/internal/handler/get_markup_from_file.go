package handler

import (
	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"golang.org/x/sync/errgroup"
	"io"
	"mime/multipart"
	"net/http"

	"github.com/gabriel-vasile/mimetype"
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"

	"github.com/OkDenAl/text-markup-gateway/internal/handler/responses"
	"github.com/OkDenAl/text-markup-gateway/internal/repo/ml-markup/httpl"
)

var ErrInvalidFileExtension = errors.New("invalid file extension")

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
// @Success 204 {object} responses.HTTPError
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

		text, err := getDataFromFile(fileHeader)
		if err != nil {
			c.JSON(http.StatusBadRequest, responses.Error(err))
			_ = c.AbortWithError(http.StatusBadRequest, err)
			return
		}

		var (
			eg     errgroup.Group
			tokens domain.Tokens
			class  domain.Class
		)

		eg.Go(func() error {
			tokens, err = markup.GetTokensFromText(c, text)
			return err
		})

		eg.Go(func() error {
			class, err = markup.GetClassFromText(c, text)
			return err
		})

		if err = eg.Wait(); err != nil {
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

		c.JSON(http.StatusOK, domain.NewTextEntities(class.Class, tokens.Labels, tokens.Tags))
	}
}

func getDataFromFile(inputFile *multipart.FileHeader) (string, error) {
	file, err := inputFile.Open()
	if err != nil {
		return "", errors.Wrap(err, "failed to open input file")
	}

	data, err := io.ReadAll(file)
	if err != nil {
		return "", errors.Wrap(err, "failed to read input file")
	}

	if err = checkFileExtension(data); err != nil {
		return "", err
	}
	return string(data), nil
}

func checkFileExtension(data []byte) error {
	var allowed = []string{"text/plain"}

	mtype := mimetype.Detect(data)
	if !mimetype.EqualsAny(mtype.String(), allowed...) {
		return errors.Wrap(ErrInvalidFileExtension, "failed to validate input file")
	}

	return nil
}
