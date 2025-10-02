from datetime import datetime, time
from typing import Union
import numpy as np
from btc_options.analytics import Result
from abc import ABC, abstractmethod
import polars as pl

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

    def set_printable(self, value: bool):
        """Enable/disable print output."""
        self._can_print = value

    def own_print(self, msg: str):
        """Print message if printing is enabled."""
        if self._can_print:
            print(msg)

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Abstract fit method to be implemented by subclasses."""
        pass

    def _convert_to_result(self, expiry: str, timestamp: datetime, const: float, coef: float, S: float, 
                          tau: float, r2_adj: float, sse: float, ) -> Result:
        """Convert regression parameters to financial rates and metrics."""
        # Calculate rates from regression parameters
        r = float(np.log(coef) / -tau)  # USD interest rate
        q = float(np.log(-const / S) / -tau)  # BTC funding rate

        return Result(
            expiry=expiry, timestamp=timestamp,
            S=S, r=r, q=q, tau=tau,
            r2=r2_adj, const=float(const), coef=float(coef), sse=float(sse),
            success_fitting=True, failure_reason=None
        )
    
    def create_results_df(self) -> pl.DataFrame:
        return pl.DataFrame(self.fit_results).with_columns(            
            [pl.col(_col).round(4) for _col in ['tau','r','q','r2','sse']]
        ).with_columns(
            (pl.col('r') - pl.col('q')).alias('r-q'),
        ).with_columns(
            (pl.col('r-q') * pl.col('tau')).round(4).alias('(r-q)*t')
        ).with_columns(
            (np.exp(pl.col('(r-q)*t')) * pl.col('S')).round(2).alias('F')
        ).with_columns(
            (pl.col('F') - pl.col('S')).alias('F-S'),
            (pl.col('F') / pl.col('S') - 1).round(4).alias('F/S-1')
        ).select(
            ['expiry','timestamp','tau','r','q','r-q','(r-q)*t','S','F','F-S','F/S-1',
             'r2','sse','success_fitting','failure_reason']
        ).join(
            self.symbol_manager.df_symbol[['expiry', 'expiry_ts']].unique(), on='expiry'
        ).sort(['timestamp','expiry_ts']).drop('expiry_ts')
    
    def get_implied_forward_price(self, result: Result) -> float:
        r, q = result['r'], result['q']
        spot, tau = result['S'], result['tau']
        return float(np.exp((r-q) * tau) * spot)
    
    def check_if_cutoff_for_0DTE(self, expiry: str, timestamp: datetime, is_0dte: bool, spot: float, tau: float) -> tuple[bool, Result|None]:
        if is_0dte:
           current_time = timestamp.time()
           if current_time >= self.cutoff_time_for_0DTE:
               # return the result for 0 base offset as it's expiring very soon
                return True, Result(expiry=expiry,
                                    timestamp=timestamp,
                                    S=spot,
                                    r=0,
                                    q=0,
                                    tau=tau,
                                    r2=np.nan,
                                    const=spot,
                                    coef=0,
                                    success_fitting=False,
                                    failure_reason="too_close_to_expiry"                                                                  
                )
        return False, None

        