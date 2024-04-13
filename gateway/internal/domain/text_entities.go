package domain

type TextEntities struct {
	Class  string   `json:"class"`
	Labels []string `json:"labels"`
	Tags   []string `json:"tags"`
}

func NewTextEntities(labels []string, tags []string) TextEntities {
	return TextEntities{Labels: labels, Tags: tags}
}
