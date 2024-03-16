package domain

type TextEntities struct {
	Class string   `json:"class"`
	Tags  []string `json:"tags"`
}

func NewTextEntities(class string, tags []string) TextEntities {
	return TextEntities{Class: class, Tags: tags}
}
