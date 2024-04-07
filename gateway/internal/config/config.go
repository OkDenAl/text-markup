package config

import (
	"fmt"
	"time"

	"github.com/go-playground/validator/v10"
	"github.com/ilyakaznacheev/cleanenv"
)

type (
	Config struct {
		Env      string       `yaml:"env" validate:"required,oneof=prod local"`
		LogLevel string       `yaml:"log_level" validate:"required"`
		HTTP     ServerConfig `yaml:"http_server" validate:"required"`
		MLClient ClientConfig `yaml:"ml_client" validate:"required"`
	}

	ServerConfig struct {
		Port           string        `yaml:"port" validate:"required" env:"HTTP_PORT"`
		ReadTimeout    time.Duration `yaml:"read_timeout" validate:"required"`
		WriteTimeout   time.Duration `yaml:"write_timeout" validate:"required"`
		SwaggerEnabled *bool         `yaml:"swagger_enabled" validate:"required"`
	}

	ClientConfig struct {
		Host    string        `yaml:"host" validate:"required"`
		Port    string        `yaml:"port" validate:"required"`
		Timeout time.Duration `yaml:"timeout" validate:"required"`
	}
)

func New(configPath string) (*Config, error) {
	cfg := &Config{}
	err := cleanenv.ReadConfig(configPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	validate := validator.New()
	if err = validate.Struct(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}
