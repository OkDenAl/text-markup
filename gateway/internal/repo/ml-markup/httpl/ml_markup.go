package httpl

import (
	"context"
	"io"
	"mime/multipart"

	"github.com/gabriel-vasile/mimetype"
	"github.com/pkg/errors"

	"github.com/OkDenAl/text-markup-gateway/internal/domain"
	"github.com/OkDenAl/text-markup-gateway/internal/handler/model"
	"github.com/OkDenAl/text-markup-gateway/pkg/logger"
)

var (
	ErrInvalidFileExtension = errors.New("invalid file extension")
	ErrInvalidData          = errors.New("invalid data")
)

type MLMarkupRepo struct {
	client iMLClient
}

func NewMLMarkupRepo(client iMLClient) (MLMarkupRepo, error) {
	return MLMarkupRepo{client: client}, nil
}

func (r MLMarkupRepo) GetEntitiesFromText(ctx context.Context, text string) (domain.TextEntities, error) {
	resp, err := r.client.GetPrediction(ctx, model.NewTextMarkupRequest(text))
	if err != nil {
		return domain.TextEntities{}, err
	}

	var te domain.TextEntities
	for i, label := range resp.Labels {
		if label != "O" {
			te.Labels = append(te.Labels, resp.Labels[i])
			te.Tags = append(te.Tags, resp.Tokens[i])
		}
	}

	if len(te.Tags) == 0 && len(te.Labels) == 0 {
		return domain.TextEntities{}, errors.Wrap(ErrInvalidData, "failed to get entities from text")
	}

	return te, nil
}

func (r MLMarkupRepo) GetEntitiesFromFile(
	ctx context.Context,
	inputFile *multipart.FileHeader,
) (domain.TextEntities, error) {
	file, err := inputFile.Open()
	if err != nil {
		return domain.TextEntities{}, errors.Wrap(err, "failed to open input file")
	}

	data, err := io.ReadAll(file)
	if err != nil {
		return domain.TextEntities{}, errors.Wrap(err, "failed to read input file")
	}

	if err = checkFileExtension(data); err != nil {
		return domain.TextEntities{}, err
	}

	log := logger.New()
	log.Debug().Str("data", string(data)).Msg("")

	resp, err := r.client.GetPrediction(ctx, model.NewTextMarkupRequest(string(data)))
	if err != nil {
		return domain.TextEntities{}, err
	}

	var te domain.TextEntities
	for i, label := range resp.Labels {
		if label != "O" {
			te.Labels = append(te.Labels, resp.Labels[i])
			te.Tags = append(te.Tags, resp.Tokens[i])
		}
	}

	if len(te.Tags) == 0 && len(te.Labels) == 0 {
		return domain.TextEntities{}, errors.Wrap(ErrInvalidData, "failed to get entities from text")
	}

	return te, nil
}

func checkFileExtension(data []byte) error {
	var allowed = []string{"text/plain", "application/pdf"}

	mtype := mimetype.Detect(data)
	if !mimetype.EqualsAny(mtype.String(), allowed...) {
		return errors.Wrap(ErrInvalidFileExtension, "failed to validate input file")
	}

	return nil
}
