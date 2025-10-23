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


COLOR_MAP = ['#9467bd', '#ff7f0e', '#e377c2', '#bcbd22', '#17becf', '#1f77b4', '#2ca02c', '#8c564b', '#d62728', '#7f7f7f']


class PlotlyManager:
    """Manager for creating interactive visualizations of options analytics results."""
    
    def __init__(self, date_str: str, future_expiries: list[str]):
        self.date_str = date_str
        self.future_expiries = future_expiries

    def _safe_show(self, fig):
        """Safely display plotly figure inline within the notebook."""
        try:
            pio.renderers.default = "notebook"
            display(HTML(fig.to_html(include_plotlyjs='cdn')))
        except:
            try:
                fig.show()
            except:
                print("✅ Plot generated successfully (display may not be visible in text environment)")

    def plot_regression_result(self, expiry: str, timestamp: str, df: pl.DataFrame, fitted_result: dict, width: int = 675, height: int = 450):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        S = df['S'][0]
        fig.add_vline(x=S, line_width=3, line_dash="dash", line_color="green", name="Spot")
        y_pred = df['strike'].to_numpy() * fitted_result['coef'] + fitted_result['const']
        fig.add_trace(go.Scatter(
            x=df['strike'], y=y_pred, mode='lines+markers', name='P-C (Fitted)',
            line=dict(color='blue'), error_y=dict(type='data', array=df['spread']/2, visible=True, color='black'),
            marker=dict(size=6, color='red', symbol='x')
        ))
        fig.add_annotation(x=0.2, xref="paper", y=1.0, yref="paper",
            text=r'$\Huge{P - C = Ke^{-r_{usd} \tau} - Se^{-r_{btc} \tau}}$', showarrow=False, font=dict(color='black'))
        fig.add_annotation(x=0.2, xref="paper", y=0.85, yref="paper",
            text=f'slope= {fitted_result["coef"]:.4f}, intercept= {fitted_result["const"]:.4f} '
                 f'r_usd= {fitted_result["r"]:.4f}, r_btc= {fitted_result["q"]:.4f}', showarrow=False, font=dict(color='black'))
        title_text = (f'<b>Weighted Least Squared (WLS) Regression:</b><br>time={timestamp}| expiry={expiry}| S={S}| '
                     f'R²={fitted_result["r2"]:.4f}| isFutExpiry={expiry in self.future_expiries}')
        fig.update_layout(title=dict(text=title_text, x=0.5, xanchor='center'), 
                          yaxis_zeroline=False, font=dict(size=10), 
                          width=width, height=height, xaxis_zeroline=False, 
                          xaxis=dict(title='Strike (K)'), yaxis=dict(title='P - C'))
        self._safe_show(fig)

    def plot_synthetic_bid_ask(self, expiry: str, timestamp: str, df: pl.DataFrame, fitted_result: dict, width: int = 600, height: int = 450, **kwargs):
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f'r = 0.0 (arb could arise across the strike for back date)', f'r = {fitted_result["r"]:.4f}'), shared_yaxes=True)
        rates = [0.0, fitted_result['r']]
        for i, rate in enumerate(rates):
            col = i + 1
            df = df.with_columns(
                (-pl.col('ask')*np.exp(rate*pl.col('tau')) + pl.col('strike')).alias('synthetic_bid'),
                (-pl.col('bid')*np.exp(rate*pl.col('tau')) + pl.col('strike')).alias('synthetic_ask')
            )
            fig.add_trace(go.Scatter(x=df['strike'], y=df['synthetic_bid'], mode='markers+lines', name='Synthetic Bid',
                line=dict(color='blue'), marker=dict(size=6, color='red', symbol='x'), showlegend=(i == 0)), row=1, col=col)
            fig.add_trace(go.Scatter(x=df['strike'], y=df['synthetic_ask'], mode='markers+lines', name='Synthetic Ask',
                line=dict(color='blue'), marker=dict(size=6, color='red', symbol='x'), showlegend=(i == 0)), row=1, col=col)
            if 'bid_price_fut' in df.columns and 'ask_price_fut' in df.columns:
                fig.add_trace(go.Scatter(x=df['strike'], y=kwargs.get('bid_price_fut', df['bid_price_fut']), mode='lines', name='Future Lower Bound',
                    line=dict(color='black', dash='dot', width=1), showlegend=(i == 0)), row=1, col=col)
                fig.add_trace(go.Scatter(x=df['strike'], y=kwargs.get('ask_price_fut', df['ask_price_fut']), mode='lines', name='Future Upper Bound',
                    line=dict(color='black', dash='dot', width=1), showlegend=(i == 0)), row=1, col=col)
            if i == 1:
                fig.add_trace(go.Scatter(x=df['strike'], y=np.full(len(df['strike']), fitted_result['F']), mode='lines', name='Fitted Forward Price',
                    line=dict(color='green'), showlegend=(i == 0)), row=1, col=col)
        fig.update_layout(title=dict(text=f'<b>Implied Forward from FV(C-P) + K <br>time={timestamp}| expiry={expiry}| S={df["S"][0]}| '
                                     f'fitted r_usd={fitted_result["r"]:.4f}</b>', x=0.5, xanchor='center'),
            xaxis=dict(title='Strike (K)'), xaxis2=dict(title='Strike (K)'), yaxis=dict(title='Forward Price', range=[df['synthetic_bid'].median()*0.975, df['synthetic_ask'].median()*1.025]),
            width=width*2, height=height, legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5))
        self._safe_show(fig)

    def plot_time_series_base_offset_analysis(self, df_results: pl.DataFrame, sorted_expiries: list, date_str: str, weight: float) -> None:
        # Create 4-panel plot 📈
        fig = make_subplots(rows=4, cols=1, 
                            subplot_titles=['USD Rate (r) %', 'BTC Rate (q) %', 'Rate Spread (r-q) %', 'Forward/Spot Ratio'], 
                            vertical_spacing=0.08, shared_xaxes=True)
                
        # Plot each expiry 🎨
        for i, exp in enumerate(sorted_expiries):
            exp_data = df_results.filter(pl.col('expiry') == exp).sort('timestamp')
            if exp_data.is_empty(): continue            
            panels_data = [(exp_data['r'] * 100).to_list(), (exp_data['q'] * 100).to_list(),
                        (exp_data['smoothened_r-q'] * 100).to_list(), (exp_data['F'] / exp_data['S']).to_list()]
            
            for panel, data in enumerate(panels_data, 1):
                fig.add_trace(go.Scatter(x=exp_data['timestamp'], y=data, mode='lines+markers', name=exp,
                    line=dict(width=2, color=COLOR_MAP[i % len(COLOR_MAP)]), marker=dict(size=4),
                    showlegend=(panel == 1), legendgroup=exp), row=panel, col=1)
            
                # Add raw spread as dotted line for comparison in panel 3
                if panel == 3:
                    fig.add_trace(go.Scatter(
                        x=exp_data['timestamp'], y=(exp_data['r-q']*100).to_list(), 
                        line=dict(dash="dot", color="gray", width=1),
                        name=f"{exp} (raw)", legendgroup=exp, showlegend=False,
                        hovertemplate="Raw rate spread: %{y:.3f}%"
                    ), row=panel, col=1)
        
        # Layout and show 🎪
        title_method = "Constrained Optimization" if (use_constrained_optimization := True) else "WLS Regression"
        fig.update_layout(title=f"<b>{date_str} Time-Series of Implied Rates and Forward/Spot Ratios - {title_method} (λ={weight})</b>",
            height=800, template='plotly_white')
        
        for i, label in enumerate(['USD Rate %', 'BTC Rate %', 'Rate Spread %', 'F/S Ratio'], 1):
            fig.update_yaxes(title_text=label, row=i, col=1)
        fig.update_xaxes(title_text="Time", row=4, col=1)
        
        self._safe_show(fig)
        print("✅ Visualization complete!")
