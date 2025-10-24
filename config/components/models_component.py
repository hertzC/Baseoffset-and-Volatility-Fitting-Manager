#!/usr/bin/env python3
"""
Models Configuration Component

Handles the 'models' section of configuration, including model selection,
parameters, bounds, and model-specific settings.
"""

from typing import Dict, Any, List, Tuple, Optional
from .base_component import BaseConfigComponent


class ModelsComponent(BaseConfigComponent):
    """
    Configuration component for models-related settings.
    
    Provides specialized access to models configuration including:
    - Model selection and enabling/disabling
    - Parameter bounds and initial values
    - Model-specific settings
    - Model comparison settings
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize models component."""
        super().__init__(config_data, 'models')
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a specific model is enabled."""
        return self.get(f'enabled_models.{model_name}', True)
    
    @property
    def wing_model_enabled(self) -> bool:
        """Check if Traditional Wing Model is enabled."""
        return self.is_model_enabled('wing_model')
    
    @property
    def time_adjusted_wing_model_enabled(self) -> bool:
        """Check if Time-Adjusted Wing Model is enabled."""
        return self.is_model_enabled('time_adjusted_wing_model')
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names."""
        enabled_models = []
        models_config = self.get('enabled_models', {})
        for model_name, enabled in models_config.items():
            if enabled:
                enabled_models.append(model_name)
        return enabled_models
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.get(model_name, {})
    
    def get_initial_params(self, model_name: str) -> Dict[str, float]:
        """Get initial parameters for a model."""
        return self.get(f'{model_name}.initial_params', {})
    
    def get_parameter_bounds(self, model_name: str) -> List[Tuple[float, float]]:
        """Get parameter bounds for a model as list of tuples for optimization."""
        bounds_dict = self.get(f'{model_name}.bounds', {})
        
        if isinstance(bounds_dict, list):
            # Handle list format (legacy)
            return [tuple(float(x) for x in bounds) for bounds in bounds_dict]
        elif isinstance(bounds_dict, dict):
            # Handle dictionary format - convert to ordered list of tuples for the model
            param_order = ['vr', 'sr', 'pc', 'cc', 'dc', 'uc', 'dsm', 'usm']
            bounds_list = []
            for param in param_order:
                if param in bounds_dict:
                    bounds_list.append(tuple(float(x) for x in bounds_dict[param]))
            return bounds_list
        else:
            return []
    
    def get_parameter_bounds_dict(self, model_name: str) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for a model as dictionary."""
        bounds_dict = self.get(f'{model_name}.bounds', {})
        
        if isinstance(bounds_dict, dict):
            return {param: tuple(float(x) for x in bounds) for param, bounds in bounds_dict.items()}
        else:
            # Convert list format to dict with default parameter names
            param_order = ['vr', 'sr', 'pc', 'cc', 'dc', 'uc', 'dsm', 'usm']
            if isinstance(bounds_dict, list) and len(bounds_dict) <= len(param_order):
                return {param_order[i]: tuple(float(x) for x in bounds) 
                       for i, bounds in enumerate(bounds_dict)}
            return {}
    
    # Time-Adjusted Wing Model specific settings
    
    @property
    def use_norm_term(self) -> bool:
        """Get whether to use normalization term for time-adjusted wing model."""
        return self.get('time_adjusted_wing_model.use_norm_term', True)
    
    # Wing Model specific settings
    
    def get_wing_model_settings(self) -> Dict[str, Any]:
        """Get wing model specific settings."""
        return {
            'enabled': self.wing_model_enabled,
            'initial_params': self.get_initial_params('wing_model'),
            'bounds': self.get_parameter_bounds_dict('wing_model')
        }
    
    def get_time_adjusted_wing_model_settings(self) -> Dict[str, Any]:
        """Get time-adjusted wing model specific settings."""
        return {
            'enabled': self.time_adjusted_wing_model_enabled,
            'use_norm_term': self.use_norm_term,
            'initial_params': self.get_initial_params('time_adjusted_wing_model'),
            'bounds': self.get_parameter_bounds_dict('time_adjusted_wing_model')
        }
    
    def get_model_comparison_settings(self) -> Dict[str, Any]:
        """Get settings for model comparison."""
        return self.get('comparison', {
            'enabled': True,
            'metrics': ['rmse', 'r_squared', 'max_error'],
            'save_results': True
        })
    
    def get_all_model_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get settings for all models."""
        settings = {}
        
        if 'wing_model' in self._section_data:
            settings['wing_model'] = self.get_wing_model_settings()
        
        if 'time_adjusted_wing_model' in self._section_data:
            settings['time_adjusted_wing_model'] = self.get_time_adjusted_wing_model_settings()
        
        return settings
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all models configuration settings."""
        return {
            'enabled_models': self.get_enabled_models(),
            'model_settings': self.get_all_model_settings(),
            'comparison': self.get_model_comparison_settings()
        }