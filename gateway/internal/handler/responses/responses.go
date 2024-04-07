package responses

type HTTPError struct {
	Error string `json:"error"`
}

func Error(err error) HTTPError {
	return HTTPError{Error: err.Error()}
}
