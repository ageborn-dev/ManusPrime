# plugins/utility/input_validator.py
import re
import logging
from typing import Dict, List, Tuple, ClassVar, Optional, Any

from plugins.base import Plugin, PluginCategory

logger = logging.getLogger("manusprime.plugins.input_validator")

class InputValidatorPlugin(Plugin):
    """Validates and sanitizes input before processing."""
    
    name: ClassVar[str] = "input_validator"
    description: ClassVar[str] = "Validates and sanitizes user input to improve security and quality"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.UTILITY
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the input validator plugin."""
        super().__init__(config)
        
        # Default rules
        self.default_rules = {
            "max_length": self.config.get("max_length", 32000),
            "min_length": self.config.get("min_length", 3),
            "check_injection": self.config.get("check_injection", True),
            "normalize_whitespace": self.config.get("normalize_whitespace", True),
            "allow_code_blocks": self.config.get("allow_code_blocks", True),
            "allow_urls": self.config.get("allow_urls", True),
            "allow_markdown": self.config.get("allow_markdown", True),
            "allowed_languages": self.config.get("allowed_languages", ["en"]),
            "profanity_filter": self.config.get("profanity_filter", False),
        }
        
        # Injection patterns to detect potential attacks
        self.injection_patterns = [
            r"ignore previous instructions",
            r"ignore all previous commands",
            r"disregard your previous instructions",
            r"forget your previous instructions",
            r"you are now",
            r"all of your instructions"
        ]
    
    async def initialize(self) -> bool:
        """Initialize the plugin with necessary setup.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        logger.info("Initializing InputValidatorPlugin")
        
        try:
            # Add any necessary initialization logic here
            # For now, we'll just do a basic check of our configuration
            
            if "max_length" in self.config and not isinstance(self.config["max_length"], int):
                logger.error("Configuration error: max_length must be an integer")
                return False
                
            if "min_length" in self.config and not isinstance(self.config["min_length"], int):
                logger.error("Configuration error: min_length must be an integer")
                return False
            
            # We'll consider the plugin initialized successfully
            logger.info("InputValidatorPlugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing InputValidatorPlugin: {str(e)}")
            return False
    
    async def execute(
        self, 
        input_text: str, 
        validation_rules: Optional[Dict] = None,
        model: Optional[str] = None,
        context: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Validate and sanitize input based on rules.
        
        Args:
            input_text: The text to validate
            validation_rules: Optional custom rules (overrides defaults)
            model: Optional target model name for model-specific validations
            context: Optional context about the request
            
        Returns:
            Dict: Validation results with sanitized input
        """
        # Combine default rules with custom rules
        rules = {**self.default_rules}
        if validation_rules:
            rules.update(validation_rules)
            
        # Track validation issues
        violations = []
        warnings = []
        sanitized = input_text
        
        # Apply basic validations
        if not sanitized or len(sanitized.strip()) < rules["min_length"]:
            violations.append("Input is too short or empty")
            
        if len(sanitized) > rules["max_length"]:
            sanitized = sanitized[:rules["max_length"]]
            warnings.append(f"Input was truncated to {rules['max_length']} characters")
        
        # Apply whitespace normalization if enabled
        if rules["normalize_whitespace"]:
            sanitized = self._normalize_whitespace(sanitized)
        
        # Check for potential security issues if enabled
        if rules["check_injection"]:
            sanitized, injection_issues = self._check_injections(sanitized)
            violations.extend(injection_issues)
        
        # Apply profanity filter if enabled
        if rules["profanity_filter"]:
            sanitized, profanity_issues = self._filter_profanity(sanitized)
            warnings.extend(profanity_issues)
        
        # Model-specific validations
        if model:
            model_issues = self._validate_for_model(sanitized, model)
            warnings.extend(model_issues)
        
        return {
            "sanitized_input": sanitized,
            "violations": violations,  # Critical issues
            "warnings": warnings,      # Non-critical issues
            "is_valid": len(violations) == 0,
            "original_length": len(input_text),
            "sanitized_length": len(sanitized)
        }
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _check_injections(self, text: str) -> Tuple[str, List[str]]:
        """Check for potential prompt injection patterns."""
        issues = []
        
        # Check for common injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Potential prompt injection detected: '{pattern}'")
        
        # Look for unusual pattern density (potential jailbreaks)
        unusual_patterns = [
            (r'\[.*?\]', 10, 'Too many bracketed sections'),
            (r'\{.*?\}', 10, 'Too many curly brace sections'),
            (r'<.*?>', 20, 'Too many HTML-like tags'),
            (r'#+', 15, 'Excessive use of hash symbols'),
            (r'system:|user:|assistant:', 5, 'Excessive use of role markers')
        ]
        
        for pattern, threshold, message in unusual_patterns:
            if len(re.findall(pattern, text)) > threshold:
                issues.append(message)
        
        # We're not modifying the text, just flagging issues
        return text, issues
    
    def _filter_profanity(self, text: str) -> Tuple[str, List[str]]:
        """Basic profanity filtering."""
        # This would normally use a proper profanity library
        # For this example, we'll use a minimal approach
        basic_profanity = ["profanity1", "profanity2", "profanity3"]
        issues = []
        
        for word in basic_profanity:
            if re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
                text = re.sub(r'\b' + re.escape(word) + r'\b', '[filtered]', text, flags=re.IGNORECASE)
                issues.append("Profanity was filtered from input")
        
        return text, issues
    
    def _validate_for_model(self, text: str, model: str) -> List[str]:
        """Model-specific validations."""
        issues = []
        
        # Check for model-specific limitations
        if "gpt-3.5" in model and len(text) > 12000:
            issues.append(f"Input may be too long for {model}")
            
        if "claude-instant" in model and "```" in text:
            issues.append(f"Multiple code blocks may cause issues with {model}")
        
        return issues