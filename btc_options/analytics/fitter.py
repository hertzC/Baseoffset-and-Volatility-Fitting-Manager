from datetime import datetime, time
from typing import Union
import numpy as np
from btc_options.analytics import Result
from abc import ABC, abstractmethod
import polars as pl

from btc_options.analytics.maths import convert_paramter_into_rate
from btc_options.data_managers.deribit_md_manager import DeribitMDManager
from btc_options.data_managers.orderbook_deribit_md_manager import OrderbookDeribitMDManager


class Fitter(ABC):
    """ Base Class for BaseOffset Fitter """

    def __init__(self, 
                 symbol_manager: Union[DeribitMDManager, OrderbookDeribitMDManager], 
                 minimum_strikes: int = 5, 
                 cutoff_time_for_0DTE: time = time(hour=4)):
        self._can_print = False
        self.minimum_strikes = minimum_strikes
        self.fit_results: list[Result] = []
        self.cutoff_time_for_0DTE = cutoff_time_for_0DTE
        self.symbol_manager = symbol_manager
        # Store original parameters for reset functionality
        self._original_params = {}

    def set_printable(self, value: bool):
        """Enable/disable print output."""
        self._can_print = value

    def own_print(self, msg: str):
        """Print message if printing is enabled."""
        if self._can_print:
            print(msg)

    def clear_results(self):
        """Clear all stored fit results."""
        self.fit_results.clear()
        print(f"✅ Cleared {self.__class__.__name__} fit results")

    def get_results_count(self) -> int:
        """Get the number of stored fit results."""
        return len(self.fit_results)

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Abstract fit method to be implemented by subclasses."""
        pass

    def _convert_to_result(self, expiry: str, timestamp: datetime, fitted_parameter: np.ndarray, S: float, 
                          tau: float, r2_adj: float, sse: float, ) -> Result:
        """Convert regression parameters to financial rates and metrics."""        
        r, q = convert_paramter_into_rate(fitted_parameter, S, tau)

        return Result(
            expiry=expiry, timestamp=timestamp,
            S=S, r=float(r), q=float(q), tau=tau,
            r2=r2_adj, const=float(fitted_parameter[0]), coef=float(fitted_parameter[1]), sse=float(sse),
            success_fitting=True, failure_reason=''
        )

    def get_implied_forward_price(self, result: Result) -> float:
        return float(np.exp((result['r']-result['q'])*result['tau'])*result['S'])
    
    def check_if_cutoff_for_0DTE(self, expiry: str, timestamp: datetime, is_0dte: bool, spot: float, tau: float) -> tuple[bool, Result|None]:
        if is_0dte:
           current_time = timestamp.time()
           if current_time >= self.cutoff_time_for_0DTE:
               # return the result for 0 base offset as it's expiring very soon
                return True, Result(expiry=expiry,
                                    timestamp=timestamp,
                                    S=spot,
                                    r=0.0,
                                    q=0.0,
                                    tau=tau,
                                    r2=np.nan,
                                    const=spot,
                                    coef=0.0,
                                    success_fitting=False,
                                    failure_reason="too_close_to_expiry"                                                                  
                )
        return False, None

    def reset_parameters(self):
        """Reset all parameters to their original initialization values."""
        for param_name, original_value in self._original_params.items():
            setattr(self, param_name, original_value)
        print("✅ Reset WLSRegressor parameters to original values")

    def update_parameters(self, **kwargs):
        """
        Update regressor parameters dynamically.
        """
        updated = []
        
        for param_name, new_value in kwargs.items():
            if param_name in set(self._original_params.keys()):
                old_value = getattr(self, param_name)
                setattr(self, param_name, new_value)
                updated.append(f"{param_name}: {old_value} → {new_value}")
            else:
                print(f"⚠️ Warning: '{param_name}' is not a valid parameter")
        
        if updated:
            print("✅ Updated parameters:")
            for update in updated:
                print(f"   {update}")
        else:
            print("❌ No valid parameters were updated")

    def get_current_parameters(self) -> dict:
        """Get current parameter values as a dictionary."""
        return {param: getattr(self, param) for param in self._original_params.keys()}

    def print_parameters(self):
        """Print current parameter values in a formatted way."""
        print("📊 Current WLSRegressor Parameters:")
        print("-" * 40)
        current_params = self.get_current_parameters()
        for param_name, value in current_params.items():
            original_value = self._original_params[param_name]
            changed = "✓" if value != original_value else " "
            print(f"  {changed} {param_name:<20}: {value}")
        print("-" * 40)