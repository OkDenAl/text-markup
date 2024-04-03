package config

import (
	"fmt"
	"github.com/go-playground/validator/v10"
	"github.com/ilyakaznacheev/cleanenv"
)

type (
	Config struct {
		Server `yaml:"server" validate:"required"`
	}
	Server struct {
		Host string `yaml:"host" validate:"required" env:"HTTP_HOST"`
		Port string `yaml:"port" validate:"required" env:"HTTP_PORT"`
	}
)

func New(configPath string) (*Config, error) {
	cfg := &Config{}
	err := cleanenv.ReadConfig(configPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("error reading config - %w", err)
	}

	err = cleanenv.UpdateEnv(cfg)
	if err != nil {
		return nil, fmt.Errorf("error updating env - %w", err)
	}

	validate := validator.New()
	if err = validate.Struct(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}
