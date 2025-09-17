"""
Configuration management for Webpage Design Analyzer
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Main application configuration"""
    
    # App settings
    app_title: str = "Webpage Design Analyzer"
    app_icon: str = "🎨"
    page_layout: str = "wide"
    
    # Analysis settings
    default_model: str = "llava:13b"
    default_temperature: float = 0.7
    default_max_tokens: int = 1000
    analysis_timeout: int = 120
    
    # Image processing
    max_image_size: tuple = (1920, 1080)
    supported_formats: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg", "webp"])
    max_file_size_mb: int = 10
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    max_history_items: int = 20
    
    # UI settings
    show_advanced_options: bool = True
    enable_batch_processing: bool = True
    show_model_info: bool = True
    
    # Storage settings
    results_directory: str = "analysis_results"
    enable_result_saving: bool = True
    auto_save_results: bool = False
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create config from environment variables"""
        config = cls()
        
        # Override with environment variables if they exist
        config.default_model = os.getenv('ANALYZER_DEFAULT_MODEL', config.default_model)
        config.default_temperature = float(os.getenv('ANALYZER_TEMPERATURE', config.default_temperature))
        config.default_max_tokens = int(os.getenv('ANALYZER_MAX_TOKENS', config.default_max_tokens))
        config.analysis_timeout = int(os.getenv('ANALYZER_TIMEOUT', config.analysis_timeout))
        config.max_file_size_mb = int(os.getenv('ANALYZER_MAX_FILE_SIZE', config.max_file_size_mb))
        config.enable_caching = os.getenv('ANALYZER_ENABLE_CACHING', 'true').lower() == 'true'
        config.log_level = os.getenv('ANALYZER_LOG_LEVEL', config.log_level)
        config.log_file = os.getenv('ANALYZER_LOG_FILE', config.log_file)
        
        return config
    
    @classmethod
    def from_file(cls, filepath: str) -> 'AppConfig':
        """Load config from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Configuration loaded from {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {str(e)}")
            return cls()  # Return default config
    
    def save_to_file(self, filepath: str) -> bool:
        """Save config to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert to dict and save
            config_dict = {
                key: getattr(self, key) 
                for key in self.__dataclass_fields__.keys()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {str(e)}")
            return False
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate configuration settings"""
        errors = {
            'errors': [],
            'warnings': []
        }
        
        # Validate temperature range
        if not 0.0 <= self.default_temperature <= 2.0:
            errors['errors'].append("Temperature must be between 0.0 and 2.0")
        
        # Validate max tokens
        if self.default_max_tokens < 10:
            errors['errors'].append("Max tokens must be at least 10")
        elif self.default_max_tokens > 4000:
            errors['warnings'].append("Max tokens is very high, may cause slow responses")
        
        # Validate timeout
        if self.analysis_timeout < 30:
            errors['warnings'].append("Analysis timeout is quite low, may cause premature timeouts")
        
        # Validate file size
        if self.max_file_size_mb > 50:
            errors['warnings'].append("Max file size is very large, may cause performance issues")
        
        # Validate supported formats
        valid_formats = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'gif']
        for fmt in self.supported_formats:
            if fmt.lower() not in valid_formats:
                errors['warnings'].append(f"Unsupported image format: {fmt}")
        
        return errors

# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def load_config(config_file: Optional[str] = None) -> AppConfig:
    """Load configuration from file or environment"""
    
    # Try to load from file first
    if config_file and os.path.exists(config_file):
        config = AppConfig.from_file(config_file)
    else:
        # Try default config file locations
        config_paths = [
            'config.json',
            'configs/app_config.json',
            os.path.expanduser('~/.analyzer/config.json')
        ]
        
        config = None
        for path in config_paths:
            if os.path.exists(path):
                config = AppConfig.from_file(path)
                break
        
        # Fall back to environment variables
        if config is None:
            config = AppConfig.from_env()
    
    # Validate configuration
    validation = config.validate()
    if validation['errors']:
        logger.error(f"Configuration errors: {validation['errors']}")
    if validation['warnings']:
        logger.warning(f"Configuration warnings: {validation['warnings']}")
    
    return config

def update_config(**kwargs) -> None:
    """Update global configuration with new values"""
    global _config
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.info(f"Config updated: {key} = {value}")
        else:
            logger.warning(f"Unknown config key: {key}")

# Prompt templates configuration
PROMPT_TEMPLATES = {
    "comprehensive": """
Analyze this webpage screenshot comprehensively and return results in JSON format:

{
  "overall_score": "score out of 10",
  "style": "design style description (e.g., minimalistic, professional, cyberpunk)",
  "color_scheme": {
    "primary_colors": ["color1", "color2"],
    "contrast_rating": "rating out of 10",
    "accessibility": "color accessibility assessment"
  },
  "typography": {
    "font_hierarchy": "assessment of font hierarchy",
    "readability": "readability score and issues",
    "consistency": "font consistency evaluation"
  },
  "layout": {
    "structure": "layout structure assessment",
    "spacing": "white space and padding evaluation",
    "alignment": "element alignment review",
    "responsive_indicators": "signs of responsive design"
  },
  "user_experience": {
    "navigation": "navigation clarity and accessibility",
    "visual_hierarchy": "information hierarchy effectiveness",
    "call_to_actions": "CTA visibility and effectiveness"
  },
  "errors": ["list of design issues and problems"],
  "improvements": ["detailed improvement suggestions"],
  "accessibility_issues": ["accessibility concerns"],
  "modern_design_score": "how modern/current the design appears"
}

Provide detailed, actionable feedback in each category.
""",

    "quick_scan": """
Provide a quick analysis of this webpage screenshot in JSON format:

{
  "quick_score": "overall score out of 10",
  "main_strengths": ["top 3 design strengths"],
  "main_issues": ["top 3 design issues"],
  "priority_fixes": ["most important improvements needed"],
  "style_category": "design style classification"
}

Focus on the most important aspects only.
""",

    "accessibility_focus": """
Analyze this webpage for accessibility and inclusive design in JSON format:

{
  "accessibility_score": "score out of 10",
  "color_contrast": "contrast ratio assessment",
  "text_readability": "text size and readability",
  "navigation_clarity": "navigation accessibility",
  "accessibility_violations": ["WCAG guideline violations"],
  "inclusive_design_suggestions": ["suggestions for better accessibility"],
  "screen_reader_considerations": ["screen reader compatibility issues"]
}

Focus primarily on accessibility and inclusive design principles.
""",

    "mobile_responsive": """
Analyze this webpage for mobile responsiveness and design in JSON format:

{
  "mobile_score": "mobile design score out of 10",
  "responsive_design": "assessment of responsive design elements",
  "mobile_navigation": "mobile navigation evaluation",
  "touch_targets": "touch target size and spacing assessment",
  "mobile_readability": "text readability on mobile",
  "mobile_issues": ["mobile-specific design problems"],
  "mobile_improvements": ["suggestions for better mobile experience"]
}

Focus on mobile user experience and responsive design principles.
"""
}

def get_prompt_template(template_name: str) -> str:
    """Get a prompt template by name"""
    return PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["comprehensive"])

def get_available_templates() -> List[str]:
    """Get list of available prompt templates"""
    return list(PROMPT_TEMPLATES.keys())

# Model configuration
MODEL_CONFIGS = {
    "llava:7b": {
        "name": "LLaVA 7B",
        "description": "Smaller, faster model good for quick analysis",
        "recommended_for": ["Quick scans", "Batch processing"],
        "estimated_vram": "8GB",
        "speed": "Fast"
    },
    "llava:13b": {
        "name": "LLaVA 13B", 
        "description": "Balanced model with good accuracy and reasonable speed",
        "recommended_for": ["Comprehensive analysis", "General use"],
        "estimated_vram": "16GB",
        "speed": "Medium"
    },
    "llava:34b": {
        "name": "LLaVA 34B",
        "description": "Large model with highest accuracy but slower",
        "recommended_for": ["Detailed analysis", "Professional use"],
        "estimated_vram": "32GB",
        "speed": "Slow"
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, {
        "name": model_name,
        "description": "Custom model",
        "recommended_for": ["General use"],
        "estimated_vram": "Unknown",
        "speed": "Unknown"
    })