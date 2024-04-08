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
		Host           string        `yaml:"host" validate:"required"`
		Port           string        `yaml:"port" validate:"required" env:"HTTP_PORT"`
		ReadTimeout    time.Duration `yaml:"read_timeout" validate:"required"`
		WriteTimeout   time.Duration `yaml:"write_timeout" validate:"required"`
		SwaggerEnabled *bool         `yaml:"swagger_enabled" validate:"required"`
	}

	ClientConfig struct {
		Host           string               `yaml:"host" validate:"required"`
		Port           string               `yaml:"port" validate:"required"`
		Timeout        time.Duration        `yaml:"timeout" validate:"required"`
		CircuitBreaker CircuitBreakerConfig `yaml:"circuit_breaker" validate:"required"`
	}

	CircuitBreakerConfig struct {
		HalfOpenMaxSuccesses int64         `yaml:"half_open_max_successes" validate:"required, gt=0"`
		MinThreshold         int64         `yaml:"min_threshold" validate:"required, gt=0"`
		FailureRate          float64       `yaml:"failure_rate" validate:"required, gt=0"`
		CounterResetInterval time.Duration `yaml:"counter_reset_interval" validate:"required"`
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
