package model

type TextKeywordsRequest struct {
	Text         string `json:"text"`
	KeywordCount int    `json:"keyword_count"`
}

func NewTextKeywordsRequest(text string, topN int) TextKeywordsRequest {
	return TextKeywordsRequest{Text: text, KeywordCount: topN}
}
