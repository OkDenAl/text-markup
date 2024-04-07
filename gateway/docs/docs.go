// Package docs GENERATED BY THE COMMAND ABOVE; DO NOT EDIT
// This file was generated by swaggo/swag
package docs

import "github.com/swaggo/swag"

const docTemplate = `{
    "schemes": {{ marshal .Schemes }},
    "swagger": "2.0",
    "info": {
        "description": "{{escape .Description}}",
        "title": "{{.Title}}",
        "contact": {
            "name": "text-markup",
            "url": "https://github.com/OkDenAl/text-markup"
        },
        "version": "{{.Version}}"
    },
    "host": "{{.Host}}",
    "basePath": "{{.BasePath}}",
    "paths": {
        "/markup": {
            "post": {
                "description": "get markup from text (string)",
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "Markup"
                ],
                "summary": "get markup from text (string)",
                "parameters": [
                    {
                        "description": "Text JSON",
                        "name": "text",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/model.TextMarkupRequest"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/domain.TextEntities"
                        }
                    },
                    "500": {
                        "description": "Internal Server Error",
                        "schema": {
                            "$ref": "#/definitions/responses.HTTPError"
                        }
                    }
                }
            }
        }
    },
    "definitions": {
        "domain.TextEntities": {
            "type": "object",
            "properties": {
                "class": {
                    "type": "string"
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            }
        },
        "model.TextMarkupRequest": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string"
                }
            }
        },
        "responses.HTTPError": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string"
                }
            }
        }
    }
}`

// SwaggerInfo holds exported Swagger Info so clients can modify it
var SwaggerInfo = &swag.Spec{
	Version:          "1.0",
	Host:             "",
	BasePath:         "/api/v1",
	Schemes:          []string{},
	Title:            "Text Markup Service",
	Description:      "Text markup - it is the service for getting markup from text.",
	InfoInstanceName: "swagger",
	SwaggerTemplate:  docTemplate,
}

func init() {
	swag.Register(SwaggerInfo.InstanceName(), SwaggerInfo)
}
