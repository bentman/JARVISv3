"""
Validation pipeline for JARVISv3 - implements pluggable validation gates.
"""
import logging
from typing import Dict, Any, List, Optional
from ..context.schemas import TaskContext
from .base import BaseValidator, ValidationResult
from .security_validator import SecurityValidator
from .budget_validator import BudgetValidator
import ast
import re
import json

logger = logging.getLogger(__name__)

class CodeValidator:
    """Helper class for code-specific validations"""
    
    @staticmethod
    def validate_python_code(code_string: str) -> ValidationResult:
        errors = []
        warnings = []
        try:
            ast.parse(code_string)
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors)
            
        dangerous_patterns = [
            (r'import\s+os', "Use of 'os' module detected"),
            (r'exec\s*\(', "Use of 'exec' is dangerous"),
            (r'eval\s*\(', "Use of 'eval' is dangerous")
        ]
        
        for pattern, warning in dangerous_patterns:
            if re.search(pattern, code_string, re.IGNORECASE):
                warnings.append(warning)
                
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

class ValidatorPipeline:
    """Main validation pipeline that manages pluggable validators"""
    
    def __init__(self):
        self.validators: List[BaseValidator] = [
            SecurityValidator(),
            BudgetValidator()
        ]
        self.code_validator = CodeValidator()
        
    def register_validator(self, validator: BaseValidator):
        """Register a new validator"""
        self.validators.append(validator)
        logger.info(f"Registered validator: {validator.name}")

    async def validate_task_context(self, context: TaskContext) -> ValidationResult:
        """Run all registered validators on the task context"""
        all_errors = []
        all_warnings = []
        is_valid = True
        
        for validator in self.validators:
            try:
                result = await validator.validate(context, context=context)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                if not result.is_valid:
                    is_valid = False
            except Exception as e:
                logger.error(f"Error in validator {validator.name}: {e}")
                all_warnings.append(f"Validator {validator.name} failed: {e}")
                
        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings
        )

    async def validate_llm_output(self, output: Dict[str, Any], context: TaskContext, expected_format: Optional[str] = None) -> ValidationResult:
        """Validate LLM output with context-aware and code-aware checks"""
        all_errors = []
        all_warnings = []
        
        # 1. Run pluggable validators on the output
        for validator in self.validators:
            res = await validator.validate(output, context=context)
            all_errors.extend(res.errors)
            all_warnings.extend(res.warnings)
            
        # 2. Code-specific validation if output contains code
        response_text = output.get("response", "")
        if isinstance(response_text, str) and ("def " in response_text or "import " in response_text):
            code_res = self.code_validator.validate_python_code(response_text)
            all_errors.extend(code_res.errors)
            all_warnings.extend(code_res.warnings)
            
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )
