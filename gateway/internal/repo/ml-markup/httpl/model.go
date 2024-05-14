package httpl

type TextKeywordsResponse struct {
	Keywords []string  `json:"keywords"`
	Scores   []float64 `json:"scores"`
}
