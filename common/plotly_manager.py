"""
Plotly Visualization Manager

Provides interactive visualization capabilities for put-call parity regression results
and synthetic option pricing analysis.
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PlotlyManager:
    """Manager for creating interactive visualizations of options analytics results."""
    
    def __init__(self, date_str: str, future_expiries: list[str]):
        """
        Initialize visualization manager.
        
        Args:
            date_str: Date string for plot titles
            future_expiries: List of futures expiries for categorization
        """
        self.date_str = date_str
        self.future_expiries = future_expiries

    def is_fut_expiry(self, expiry: str) -> bool:
        """Check if expiry corresponds to a futures contract."""
        return expiry in self.future_expiries

    def plot_regression_result(self, expiry: str, timestamp: str,
                             df_option_synthetic: pl.DataFrame, fitted_result: dict, width: int = 900, height: int = 600):
        """
        Plot regression fit of put-call parity with error bars and formula.
        
        Args:
            expiry: Option expiry
            timestamp: Analysis timestamp
            df_option_synthetic: Synthetic option data
            fitted_result: Regression results dictionary
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add spot price reference line
        S = df_option_synthetic['S'][0]
        fig.add_vline(x=S, line_width=3, line_dash="dash", 
                     line_color="green", name="Spot")
        
        # Calculate fitted line
        y_pred = (df_option_synthetic['strike'].to_numpy() * fitted_result['coef'] + 
                 fitted_result['const'])

        # Add regression line with error bars
        fig.add_trace(go.Scatter(
            x=df_option_synthetic['strike'], 
            y=y_pred, 
            mode='lines+markers', 
            name='P-C (Fitted)', 
            line=dict(color='blue'),
            error_y=dict(type='data', 
                        array=df_option_synthetic['spread']/2, 
                        visible=True, 
                        color='black'), 
            marker=dict(size=6, color='red', symbol='x')
        ))

        # Add mathematical formula annotation
        annotation_text = r'$\Huge{P - C = Ke^{-r_{usd} \tau} - Se^{-r_{btc} \tau}}$'
        
        fig.add_annotation(
            x=0.2, xref="paper", y=1.0, yref="paper",
            text=annotation_text, showarrow=False, font=dict(color='black')
        )
        
        # Add parameter values
        fig.add_annotation(
            x=0.2, xref="paper", y=0.85, yref="paper",
            text=f'slope= {fitted_result["coef"]:.4f}, intercept= {fitted_result["const"]:.4f} '
                 f'r_usd= {fitted_result["r"]:.4f}, r_btc= {fitted_result["q"]:.4f}',
            showarrow=False, font=dict(color='black')
        )
        
        # Configure layout
        title_text = ('Weighted Least Squared (WLS) Regression:<br>'
                     f'date={self.date_str} expiry={expiry}, S={S}, time={timestamp}, '
                     f'R²={fitted_result["r2"]:.4f}, isFutExpiry={self.is_fut_expiry(expiry)}')

        fig.update_layout(
            title=dict(text=title_text),
            yaxis_zeroline=False, 
            font=dict(size=10),
            width=width, height=height,
            xaxis_zeroline=False, 
            xaxis=dict(title='Strike (K)'), 
            yaxis=dict(title='P - C')
        )
        fig.show()

    def plot_synthetic_bid_ask(self, expiry: str, timestamp: str,
                              df_option_synthetic: pl.DataFrame, fitted_result: dict,
                              use_fitted_rate: bool = False, width: int = 900, height: int = 600):
        """
        Plot synthetic forward prices vs futures market prices.
        
        Args:
            expiry: Option expiry
            timestamp: Analysis timestamp
            df_option_synthetic: Synthetic option data
            fitted_result: Regression results
            use_fitted_rate: Whether to use fitted USD rate for discounting
        """
        fig = go.Figure()
        
        rate = fitted_result['r'] if use_fitted_rate else 0.0

        # Calculate synthetic prices
        df = df_option_synthetic.with_columns(
            (-pl.col('ask')*np.exp(rate*pl.col('tau')) + pl.col('strike')).alias('synthetic_bid'),
            (-pl.col('bid')*np.exp(rate*pl.col('tau')) + pl.col('strike')).alias('synthetic_ask')
        )

        # Add synthetic bid/ask traces
        fig.add_trace(go.Scatter(
            x=df['strike'], y=df['synthetic_bid'], 
            mode='markers+lines', name='Synthetic Bid', 
            line=dict(color='blue'), 
            marker=dict(size=6, color='red', symbol='x')
        ))

        fig.add_trace(go.Scatter(
            x=df['strike'], y=df['synthetic_ask'], 
            mode='markers+lines', name='Synthetic Ask', 
            line=dict(color='blue'), 
            marker=dict(size=6, color='red', symbol='x')
        ))
        
        # Add futures market data if available
        if 'bid_price_fut' in df.columns and 'ask_price_fut' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['strike'], y=df['bid_price_fut'], 
                mode='lines', name='Bid Future', 
                line=dict(color='black', dash='dot', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=df['strike'], y=df['ask_price_fut'], 
                mode='lines', name='Ask Future', 
                line=dict(color='black', dash='dot', width=1)
            ))
        
        # Add fitted forward price line
        fig.add_trace(go.Scatter(
            x=df['strike'],
            y=np.full(len(df['strike']), fitted_result['F']),
            mode='lines', name='Fitted Forward Price',
            line=dict(color='green')
        ))

        fig.update_layout(
            title=dict(text=f'date={timestamp}, expiry={expiry}, '
                           f'S={df_option_synthetic["S"][0]}, r_usd={rate:.4f}'),
            xaxis=dict(title='Strike (K)'),
            yaxis=dict(title='Forward Price'),
            width=width, height=height,
        )
        fig.show()

    def plot_time_series_results(self, df_results: pl.DataFrame, metric: str = 'r-q'):
        """
        Plot time series of fitted parameters across expiries.
        
        Args:
            df_results: DataFrame with fitted results over time
            metric: Metric to plot ('r-q', 'r', 'q', 'F', etc.)
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Plot each expiry as separate trace
        for expiry in df_results['expiry'].unique():
            expiry_data = df_results.filter(pl.col('expiry') == expiry)
            fig.add_trace(go.Scatter(
                x=expiry_data["timestamp"], 
                y=expiry_data[metric], 
                name=f'{expiry}', 
                mode='lines'
            ))

        fig.update_layout(
            title=dict(text=f'{metric} across different expiries'),
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            xaxis=dict(title='Time'),
            yaxis=dict(title=metric),
            width=900, height=600,
        )
        fig.show()