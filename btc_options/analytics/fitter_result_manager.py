import numpy as np
import polars as pl

from btc_options.analytics import Result


class FitterResultManager:
    def __init__(self, option_expiries: list[str], future_expiries: list[str], symbol_df: pl.DataFrame, fit_results: list[Result], successful_fits: dict[str, dict]):
        self.opt_expiries = option_expiries
        self.future_expiries = future_expiries
        self.symbol_df = symbol_df[['expiry', 'expiry_ts']].unique()
        self.fit_results = fit_results
        self.fit_result_per_expiry: dict[str, list[Result]] = {expiry: [] for expiry in option_expiries}
        self.successful_fits = successful_fits
    
    def organize_results_by_expiry(self):
        """Organize fit results by expiry date."""
        for result in self.fit_results:
            if result['expiry'] in self.fit_result_per_expiry:
                self.fit_result_per_expiry[result['expiry']].append(result)
            else:
                self.fit_result_per_expiry[result['expiry']] = [result]

    def get_expiry_summary(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.group_by('expiry').agg([
            pl.len().alias('obs'),
            (pl.col('r') * 100).drop_nans().mean().round(2).alias('r_%'),
            (pl.col('r') * 100).min().round(2).alias('r_min%'),
            (pl.col('r') * 100).max().round(2).alias('r_max%'),
            (pl.col('q') * 100).drop_nans().mean().round(2).alias('q_%'),
            (pl.col('q') * 100).min().round(2).alias('q_min%'),
            (pl.col('q') * 100).max().round(2).alias('q_max%'),
            (pl.col('(r-q)*t') * 100).drop_nans().mean().round(2).alias('spread_%'),
            (pl.col('r-q') * 100).drop_nans().mean().round(2).alias('spread_%(pa)'),
            (pl.col('F-S')).drop_nans().mean().round(1).alias('BaseOffset_$'),
            (pl.col('F/S-1') * 100).drop_nans().mean().round(2).alias('Basis_%'),
            (pl.col('F/S-1') * 100).drop_nans().std().round(4).alias('Basis_%(stdev)'),
            # pl.col('r2').drop_nans().mean().round(4).alias('R²'),
            pl.col('sse').drop_nans().mean().round(4).alias('Avg_SSE')
        ]).join(self.symbol_df, on='expiry').sort('expiry_ts').drop('expiry_ts')

        df_successful = pl.DataFrame([{'expiry': exp, 'is_fut': exp in self.future_expiries, **values} for exp, values in self.successful_fits.items()])
        return result.join(
            df_successful,
            on='expiry'
        ).with_columns(
            (pl.col('successful') / pl.col('total') * 100).round(2).alias('success_rate_%')
        ).drop(['obs'])
    
    def get_failure_summary(self, df: pl.DataFrame) -> pl.DataFrame:
        """Get summary of failed fits by reason."""
        return df.filter(pl.col('success_fitting') == False).group_by(
            ['expiry','failure_reason']
        ).len().pivot(
            on='failure_reason', 
            values='len'
        ).join(self.symbol_df, on='expiry').sort('expiry_ts').drop('expiry_ts').fill_null(0)
    

    def create_results_df(self) -> pl.DataFrame:
        return pl.DataFrame(self.fit_results).with_columns(
            [pl.col(_col).round(4) for _col in ['tau','r','q','r2','sse']]
        ).with_columns(
            (pl.col('r') - pl.col('q')).alias('r-q')
        ).with_columns(
            ((pl.col('r') - pl.col('q')) * pl.col('tau')).round(4).alias('(r-q)*t')
        ).with_columns(
            (np.exp(pl.col('(r-q)*t')) * pl.col('S')).round(2).alias('F')
        ).with_columns(
            (pl.col('F') - pl.col('S')).alias('F-S'),
            (pl.col('F') / pl.col('S') - 1).round(4).alias('F/S-1')
        ).select(
            ['expiry','timestamp','tau','r','q','r-q','(r-q)*t','S','F','F-S','F/S-1',
             'r2','sse','success_fitting','failure_reason','const','coef']
        ).join(
            self.symbol_df, on='expiry'
        ).sort(
            ['timestamp','expiry_ts']
        ).drop('expiry_ts')
    