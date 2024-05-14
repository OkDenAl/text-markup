package domain

type Keywords struct {
	Keywords []string `json:"keywords"`
}

func NewKeywords(kws []string) Keywords {
	return Keywords{Keywords: kws}
}
