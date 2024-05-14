package domain

type TextEntities struct {
	Class    string   `json:"class"`
	Labels   []string `json:"labels"`
	Tags     []string `json:"tags"`
	Keywords []string `json:"keywords"`
}

func NewTextEntities(class string, labels, tags, kw []string) TextEntities {
	return TextEntities{Labels: labels, Tags: tags, Class: class, Keywords: kw}
}
