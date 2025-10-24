"""
Weighted Least Squares Regressor for Bitcoin options put-call parity analysis.
Extracts USD and BTC interest rates from options pricing data.
"""

from typing import Union
import polars as pl
from config.base_offset_config import BaseOffsetConfig
from config.volatility_config import VolatilityConfig
from utils.base_offset_fitter import Result
from utils.base_offset_fitter.fitter import Fitter
from utils.market_data.deribit_md_manager import DeribitMDManager
from utils.market_data.orderbook_deribit_md_manager import OrderbookDeribitMDManager
import statsmodels.api as sm


class WLSRegressor(Fitter):
    """Weighted Least Squares regressor for put-call parity analysis."""
    
    def __init__(self, symbol_manager: Union[DeribitMDManager, OrderbookDeribitMDManager],
                 config_loader: VolatilityConfig|BaseOffsetConfig):
        super().__init__(symbol_manager, config_loader)
        self.set_printable(self.config_loader.get('fitting.wls.printable', False))

    def fit(self, df: pl.DataFrame, **kwargs) -> Result:
        """
        Fit put-call parity regression to extract interest rates.
        
        Solves: P - C = K*exp(-r*t) - S*exp(-q*t)
        Linearized as: y = a*K + b, where a = exp(-r*t), b = -S*exp(-q*t)
        
        Args:
            df: Option synthetic data with columns: 'mid', 'strike', 'spread', 'S', 'tau'
            
        Returns:
            Result dictionary with fitted parameters and derived rates
        """
        if df.is_empty():
            raise ValueError("DataFrame is empty. Cannot fit model.")
        expiry = kwargs['expiry']
        timestamp = kwargs['timestamp']
        spot_value, tau = df["S"][0], df["tau"][0]

        is_cutoff, result = self.check_if_cutoff_for_0DTE(expiry, timestamp, self.symbol_manager.is_expiry_today(expiry), spot_value, tau)

        if not is_cutoff:                
            # Extract data and construct regression inputs
            y = df["mid"].to_numpy()
            X = df["strike"].to_numpy()
            weight = 1 / df["spread"].to_numpy()
            weight = weight * weight  # Square weights for WLS
            X_with_const = sm.add_constant(X)
            
            # Fit weighted least squares
            model = sm.WLS(y, X_with_const, weights=weight).fit()
            self.own_print(model.summary())
            
            # Convert parameters to financial rates
            r2_adj, sse = float(model.rsquared_adj), float(model.ssr)            

            result = self._convert_to_result(expiry, timestamp, model.params, spot_value, tau, r2_adj, sse)

        self.fit_results.append(result)        
        return result
