package domain

type FileTextEntities struct {
	Text     string   `json:"text"`
	Class    string   `json:"class"`
	Labels   []string `json:"labels"`
	Tags     []string `json:"tags"`
	Keywords []string `json:"keywords"`
}

func NewFileTextEntities(text, class string, labels, tags, kw []string) FileTextEntities {
	return FileTextEntities{Labels: labels, Tags: tags, Class: class, Keywords: kw, Text: text}
}
