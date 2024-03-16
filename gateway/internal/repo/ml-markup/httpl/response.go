package httpl

type MLResponse struct {
	Tokens []string `json:"tokens"`
	Labels []string `json:"labels"`
}
