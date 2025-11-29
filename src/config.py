"""
Configuration file for IDS ML system.
Centralizes all configuration parameters.
"""

import os
from pathlib import Path


class Config:
    """Base configuration."""
    
    # Application
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ids-ml-secret-key-2025'
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    
    # Server
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    # File Upload
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Models
    DEFAULT_MODEL = 'random_forest'
    AVAILABLE_MODELS = ['random_forest', 'svm', 'neural_network']
    
    # Alert Manager
    ALERT_CONFIG = {
        'email_enabled': os.environ.get('EMAIL_ENABLED', 'false').lower() == 'true',
        'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.environ.get('SMTP_PORT', 587)),
        'smtp_from': os.environ.get('SMTP_FROM', 'ids@example.com'),
        'smtp_to': os.environ.get('SMTP_TO', 'admin@example.com'),
        'smtp_username': os.environ.get('SMTP_USERNAME', ''),
        'smtp_password': os.environ.get('SMTP_PASSWORD', ''),
        'siem_enabled': os.environ.get('SIEM_ENABLED', 'false').lower() == 'true',
        'siem_url': os.environ.get('SIEM_URL', 'http://localhost:9200/ids-alerts/_doc'),
        'webhook_url': os.environ.get('WEBHOOK_URL', None)
    }
    
    # ELK Stack
    ELASTICSEARCH_HOSTS = os.environ.get('ELASTICSEARCH_HOSTS', 'localhost:9200').split(',')
    ELASTICSEARCH_USERNAME = os.environ.get('ELASTICSEARCH_USERNAME', None)
    ELASTICSEARCH_PASSWORD = os.environ.get('ELASTICSEARCH_PASSWORD', None)
    
    # Real-time Monitoring
    MONITORING_UPDATE_INTERVAL = 2  # seconds
    MONITORING_INTERFACE = os.environ.get('NETWORK_INTERFACE', 'eth0')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOGS_DIR / 'ids_ml.log'
    
    # Machine Learning
    ML_CONFIG = {
        'test_size': 0.2,
        'random_state': 42,
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        },
        'svm': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42
        },
        'neural_network': {
            'hidden_layers': [128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32
        }
    }
    
    # Attack Types
    ATTACK_TYPES = {
        'DoS': 'Denial of Service',
        'DDoS': 'Distributed Denial of Service',
        'Probe': 'Port Scanning/Reconnaissance',
        'R2L': 'Remote to Local Attack',
        'U2R': 'User to Root Attack',
        'Botnet': 'Botnet Activity',
        'Brute Force': 'Brute Force Attack',
        'SQL Injection': 'SQL Injection Attack',
        'XSS': 'Cross-Site Scripting',
        'Normal': 'Normal Traffic'
    }
    
    # Severity Levels
    SEVERITY_LEVELS = ['low', 'medium', 'high', 'critical']
    
    # WebSocket
    SOCKETIO_MESSAGE_QUEUE = None
    SOCKETIO_CORS_ALLOWED_ORIGINS = '*'
    
    @classmethod
    def init_directories(cls):
        """Create necessary directories."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.UPLOAD_FOLDER
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name):
        """Get path for a specific model."""
        if model_name == 'neural_network':
            return cls.MODELS_DIR / f'{model_name}_model.h5'
        else:
            return cls.MODELS_DIR / f'{model_name}_model.pkl'
    
    @classmethod
    def get_preprocessor_path(cls):
        """Get preprocessor path."""
        return cls.MODELS_DIR / 'preprocessor.pkl'


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    ALERT_CONFIG = {
        **Config.ALERT_CONFIG,
        'email_enabled': True,
        'siem_enabled': True
    }


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    
    # Use separate directories for testing
    DATA_DIR = Config.BASE_DIR / 'test_data'
    MODELS_DIR = Config.BASE_DIR / 'test_models'
    LOGS_DIR = Config.BASE_DIR / 'test_logs'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """
    Get configuration based on environment.
    
    Args:
        env: Environment name (development, production, testing)
        
    Returns:
        Config class
    """
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    
    return config.get(env, config['default'])