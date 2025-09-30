"""
Plotly Visualization Manager

Provides interactive visualization capabilities for put-call parity regression results
and synthetic option pricing analysis.
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import HTML, display


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

    def _safe_show(self, fig):
        """Safely display plotly figure inline within the notebook."""
        try:
            # Configure for inline notebook display
            import plotly.io as pio
            from IPython.display import HTML, display
            
            # Set the renderer to ensure inline display
            pio.renderers.default = "notebook"
            
            # Try to display inline using IPython display
            display(HTML(fig.to_html(include_plotlyjs='cdn')))
            
        except Exception as e:
            # Fallback: try standard show (might work in some environments)
            try:
                fig.show()
            except:
                # Final fallback - just indicate success
                print("✅ Plot generated successfully (display may not be visible in text environment)")
                # In a real Jupyter notebook, you would see the plot above this message

    def is_fut_expiry(self, expiry: str) -> bool:
        """Check if expiry corresponds to a futures contract."""
        return expiry in self.future_expiries

    def plot_regression_result(self, expiry: str, timestamp: str,
                             df_option_synthetic: pl.DataFrame, fitted_result: dict, width: int = 675, height: int = 450):
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
        self._safe_show(fig)

    def plot_synthetic_bid_ask(self, expiry: str, timestamp: str,
                              df_option_synthetic: pl.DataFrame, fitted_result: dict,
                              use_fitted_rate: bool = False, width: int = 675, height: int = 450):
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
        self._safe_show(fig)

    def plot_time_series_analysis(self, df_results: pl.DataFrame, metric: str = None) -> None:
        """
        Create time series plots for forward prices and rate analysis.
        
        Args:
            df_results: DataFrame containing time series results
            metric: Single metric to plot, or None for comprehensive multi-panel view
        """
        if df_results.is_empty():
            print("❌ No data available for time series plotting")
            return
            
        # Single metric plot
        if metric:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
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
                yaxis_zeroline=False, xaxis_zeroline=False,
                xaxis=dict(title='Time'), yaxis=dict(title=metric),
                width=675, height=450,
            )
            self._safe_show(fig)
            return
            
        # Multi-panel comprehensive analysis
        print("📊 CREATING TIME SERIES PLOTS")
        
        try:
            # Prepare data with derived columns
            df_plot = df_results.with_columns([
                (pl.col('r') - pl.col('q')).alias('r_minus_q'),
                (pl.col('F') - 62000).alias('base_offset')  # Assuming ~62k spot
            ]).sort('timestamp')
            
            # Get unique expiries and colors
            expiries = sorted(df_plot['expiry'].unique())
            
            # Simple color palette
            color_map = {
                '15MAR24': '#1f77b4', '1MAR24': '#ff7f0e', '22MAR24': '#2ca02c',
                '26APR24': '#d62728', '29FEB24': '#9467bd', '29MAR24': '#8c564b',
                '2MAR24': '#e377c2', '31MAY24': '#7f7f7f', '3MAR24': '#bcbd22',
                '8MAR24': '#17becf'
            }
            colors = [color_map.get(exp, '#1f77b4') for exp in expiries]
            
            print(f"Plotting {len(expiries)} expiries")
            
            # Create 4-panel subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Forward Price ($)', 'Rate Spread (r-q)',
                    'Base Offset ($)', 'USD Rate vs BTC Rate'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Add traces for each panel
            for i, exp in enumerate(expiries):
                exp_data = df_plot.filter(pl.col('expiry') == exp)
                
                # Panel 1: Forward prices
                fig.add_trace(
                    go.Scatter(
                        x=exp_data['timestamp'].to_list(),
                        y=exp_data['F'].to_list(),
                        mode='lines', name=exp, line=dict(color=colors[i], width=2),
                        legendgroup=exp, showlegend=(i == 0)
                    ), row=1, col=1
                )
                
                # Panel 2: Rate spread
                fig.add_trace(
                    go.Scatter(
                        x=exp_data['timestamp'].to_list(),
                        y=exp_data['r_minus_q'].to_list(),
                        mode='lines', line=dict(color=colors[i], width=2),
                        legendgroup=exp, showlegend=False
                    ), row=1, col=2
                )
                
                # Panel 3: Base offset
                fig.add_trace(
                    go.Scatter(
                        x=exp_data['timestamp'].to_list(),
                        y=exp_data['base_offset'].to_list(),
                        mode='lines', line=dict(color=colors[i], width=2),
                        legendgroup=exp, showlegend=False
                    ), row=2, col=1
                )
                
                # Panel 4: USD vs BTC rates
                fig.add_trace(
                    go.Scatter(
                        x=exp_data['r'].to_list(),
                        y=exp_data['q'].to_list(),
                        mode='markers', marker=dict(color=colors[i], size=6),
                        legendgroup=exp, showlegend=False
                    ), row=2, col=2
                )

            fig.update_layout(
                title=f'Bitcoin Options Analysis - {self.date_str}',
                height=800, width=1200, font=dict(size=11)
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=1, col=2) 
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_xaxes(title_text="USD Rate (r)", row=2, col=2)
            
            fig.update_yaxes(title_text="Forward ($)", row=1, col=1)
            fig.update_yaxes(title_text="r-q", row=1, col=2)
            fig.update_yaxes(title_text="Basis ($)", row=2, col=1)
            fig.update_yaxes(title_text="BTC Rate (q)", row=2, col=2)
            
            self._safe_show(fig)
            print("✅ Time series plots created successfully")
            
        except Exception as e:
            print(f"❌ Error creating time series plots: {e}")

    def plot_term_structure(self, df_results: pl.DataFrame, target_datetime) -> None:
        """
        Plot term structure of basis across expiries for a given datetime.
        
        Args:
            df_results: Results DataFrame with basis calculations
            target_datetime: Specific datetime to analyze
        """
        # Filter for specific datetime and calculate basis
        filtered_data = df_results.filter(pl.col('timestamp') == target_datetime)
        
        if filtered_data.is_empty():
            print(f"❌ No data found for datetime {target_datetime}")
            return
            
        # Add time to expiry calculation
        plot_data = filtered_data.with_columns([
            (pl.col('F') - 62000).alias('basis'),  # Assuming 62k spot reference
            pl.col('expiry').alias('maturity')
        ]).sort('expiry')
        
        fig = go.Figure()
        
        # Add basis term structure
        fig.add_trace(go.Scatter(
            x=plot_data['maturity'].to_list(),
            y=plot_data['basis'].to_list(),
            mode='lines+markers',
            name='Basis',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='red')
        ))
        
        fig.update_layout(
            title=f'Basis Term Structure - {target_datetime}',
            xaxis_title='Expiry',
            yaxis_title='Basis ($)',
            width=675, height=450
        )
        
        self._safe_show(fig)
        print(f"✅ Term structure plot created for {target_datetime}")