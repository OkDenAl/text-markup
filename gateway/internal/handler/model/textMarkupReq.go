package model

type TextMarkupRequest struct {
	Text string `json:"text"`
}

func NewTextMarkupRequest(text string) TextMarkupRequest {
	return TextMarkupRequest{Text: text}
}
