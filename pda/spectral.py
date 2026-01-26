"""
Spectral deconvolution functions for PDA spectrophotometer analysis.

This module provides core functions for calculating concentrations of NADH and pyruvate
from spectral absorbance data using linear regression against extinction coefficient standards.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_concentrations(
    spectrum_df,
    standards_df,
    wavelength_range=(320, 420),
    absorbance_max=2,
    fit_intercept=False,
    fixed_nadh=None,
    fixed_pyr=None,
    plot=False,
    plot_title=None
):
    """
    Calculates concentrations of NADH and Pyruvate using linear regression against standards.
    Optionally fixes one component concentration and solves for the other.
    Optionally generates an interactive Plotly visualization of the spectral decomposition.

    Args:
        spectrum_df: DataFrame containing 'Wavelength' and 'Absorbance'.
        standards_df: DataFrame containing 'Wavelength', 'NADH_Coeff', and 'PYR_Coeff'.
        wavelength_range: Tuple (min_nm, max_nm) to filter data. Default is None.
        absorbance_max: Float value to exclude absorbance readings above this limit. Default is None.
        fit_intercept: Boolean. If True, allows a non-zero intercept (baseline offset). Default is False.
        fixed_nadh: Float. If provided, fixes NADH concentration (mM) and solves for Pyruvate only.
        fixed_pyr: Float. If provided, fixes Pyruvate concentration (mM) and solves for NADH only.
        plot: Boolean. If True, generates and displays a Plotly figure. Default is False.
        plot_title: String. Custom title for the plot. If None, auto-generates title.

    Returns:
        dict: {
            'NADH_Conc': float,           # Calculated/fixed NADH concentration (mM)
            'PYR_Conc': float,            # Calculated/fixed Pyruvate concentration (mM)
            'Intercept': float,           # Baseline offset (if fit_intercept=True)
            'R_squared': float,           # Goodness of fit
            'fixed_component': str or None, # Which component was fixed ('NADH', 'PYR', or None)
            'wavelengths': np.array,      # Wavelength values used in fit
            'raw_absorbance': np.array,   # Raw absorbance data
            'fitted_absorbance': np.array, # Total fitted absorbance
            'nadh_component': np.array,   # NADH contribution to total absorbance
            'pyr_component': np.array,    # Pyruvate contribution to total absorbance
            'intercept_component': np.array, # Baseline contribution (if applicable)
            'residuals': np.array,        # Fit residuals
            'fig': plotly.graph_objects.Figure or None  # Plotly figure (if plot=True)
        }

    Notes:
        - Cannot specify both fixed_nadh and fixed_pyr simultaneously
        - When a component is fixed, R² reflects the quality of fit for the remaining component(s)
    """

    # Validate that both concentrations are not fixed simultaneously
    if fixed_nadh is not None and fixed_pyr is not None:
        raise ValueError("Cannot fix both NADH and Pyruvate concentrations simultaneously. "
                        "Specify only one or neither.")

    # Merge spectrum with standards on Wavelength
    merged_data = pd.merge(spectrum_df, standards_df, on='Wavelength', how='inner')

    # Filter by Wavelength Range if provided
    if wavelength_range:
        min_wav, max_wav = wavelength_range
        merged_data = merged_data[
            (merged_data['Wavelength'] >= min_wav) &
            (merged_data['Wavelength'] <= max_wav)
        ]

    # Filter by Absorbance Max if provided
    if absorbance_max is not None:
        merged_data = merged_data[merged_data['Absorbance'] <= absorbance_max]

    if merged_data.empty or len(merged_data) < 2:
        return {
            'NADH_Conc': None,
            'PYR_Conc': None,
            'Intercept': 0,
            'R_squared': None,
            'fixed_component': None,
            'wavelengths': None,
            'raw_absorbance': None,
            'fitted_absorbance': None,
            'nadh_component': None,
            'pyr_component': None,
            'intercept_component': None,
            'residuals': None,
            'fig': None
        }

    wavelengths = merged_data['Wavelength'].values
    y_original = merged_data['Absorbance'].values

    # Determine fitting mode
    fixed_component = None

    if fixed_nadh is not None:
        # NADH is fixed, solve for Pyruvate (and optionally intercept)
        fixed_component = 'NADH'
        nadh_conc = fixed_nadh

        # Subtract NADH contribution from observed absorbance
        nadh_contribution = fixed_nadh * merged_data['NADH_Coeff'].values
        y_adjusted = y_original - nadh_contribution

        # Fit only Pyruvate (and optionally intercept)
        X = merged_data[['PYR_Coeff']].copy()
        if fit_intercept:
            X = sm.add_constant(X)

        model = sm.OLS(y_adjusted, X).fit()
        pyr_conc = model.params.get('PYR_Coeff', 0)
        intercept = model.params.get('const', 0)
        r_squared = model.rsquared

    elif fixed_pyr is not None:
        # Pyruvate is fixed, solve for NADH (and optionally intercept)
        fixed_component = 'PYR'
        pyr_conc = fixed_pyr

        # Subtract Pyruvate contribution from observed absorbance
        pyr_contribution = fixed_pyr * merged_data['PYR_Coeff'].values
        y_adjusted = y_original - pyr_contribution

        # Fit only NADH (and optionally intercept)
        X = merged_data[['NADH_Coeff']].copy()
        if fit_intercept:
            X = sm.add_constant(X)

        model = sm.OLS(y_adjusted, X).fit()
        nadh_conc = model.params.get('NADH_Coeff', 0)
        intercept = model.params.get('const', 0)
        r_squared = model.rsquared

    else:
        # Standard mode: fit both NADH and Pyruvate
        fixed_component = None
        X = merged_data[['NADH_Coeff', 'PYR_Coeff']].copy()

        if fit_intercept:
            X = sm.add_constant(X)

        model = sm.OLS(y_original, X).fit()
        nadh_conc = model.params.get('NADH_Coeff', 0)
        pyr_conc = model.params.get('PYR_Coeff', 0)
        intercept = model.params.get('const', 0)
        r_squared = model.rsquared

    # Calculate component contributions
    nadh_component = nadh_conc * merged_data['NADH_Coeff'].values
    pyr_component = pyr_conc * merged_data['PYR_Coeff'].values
    intercept_component = np.full_like(wavelengths, intercept) if fit_intercept else np.zeros_like(wavelengths)

    # Total fitted absorbance
    fitted_absorbance = nadh_component + pyr_component + intercept_component

    # Residuals
    residuals = y_original - fitted_absorbance

    # Prepare return dictionary
    result = {
        'NADH_Conc': nadh_conc,
        'PYR_Conc': pyr_conc,
        'Intercept': intercept,
        'R_squared': r_squared,
        'fixed_component': fixed_component,
        'wavelengths': wavelengths,
        'raw_absorbance': y_original,
        'fitted_absorbance': fitted_absorbance,
        'nadh_component': nadh_component,
        'pyr_component': pyr_component,
        'intercept_component': intercept_component,
        'residuals': residuals,
        'fig': None
    }

    # Generate plot if requested
    if plot:
        fig = _create_deconvolution_plot(result, wavelength_range, plot_title)
        result['fig'] = fig
        fig.show()

    return result


def _create_deconvolution_plot(result, wavelength_range, plot_title):
    """
    Internal function to create a comprehensive Plotly visualization of spectral deconvolution.

    Args:
        result: Dictionary output from calculate_concentrations
        wavelength_range: Tuple of (min, max) wavelengths or None
        plot_title: Custom plot title or None

    Returns:
        plotly.graph_objects.Figure
    """
    wavelengths = result['wavelengths']
    fixed_component = result['fixed_component']

    # Create subplots: main spectrum plot + residuals plot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        subplot_titles=('Spectral Decomposition', 'Fit Residuals'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # --- Main Plot: Spectral Components ---

    # Raw data
    fig.add_trace(
        go.Scatter(
            x=wavelengths,
            y=result['raw_absorbance'],
            mode='markers',
            name='Raw Data',
            marker=dict(size=4, color='rgba(100, 100, 100, 0.6)', symbol='circle'),
            legendgroup='data',
            showlegend=True
        ),
        row=1, col=1
    )

    # Total fit
    fig.add_trace(
        go.Scatter(
            x=wavelengths,
            y=result['fitted_absorbance'],
            mode='lines',
            name=f'Total Fit (R² = {result["R_squared"]:.6f})',
            line=dict(color='black', width=2.5),
            legendgroup='fit',
            showlegend=True
        ),
        row=1, col=1
    )

    # NADH component (with indicator if fixed)
    nadh_label = f'NADH Component ({result["NADH_Conc"]:.4f} mM)'
    if fixed_component == 'NADH':
        nadh_label += ' [FIXED]'

    fig.add_trace(
        go.Scatter(
            x=wavelengths,
            y=result['nadh_component'],
            mode='lines',
            name=nadh_label,
            line=dict(
                color='royalblue',
                width=3 if fixed_component == 'NADH' else 2,
                dash='solid' if fixed_component == 'NADH' else 'dash'
            ),
            legendgroup='components',
            showlegend=True
        ),
        row=1, col=1
    )

    # Pyruvate component (with indicator if fixed)
    pyr_label = f'Pyruvate Component ({result["PYR_Conc"]:.4f} mM)'
    if fixed_component == 'PYR':
        pyr_label += ' [FIXED]'

    fig.add_trace(
        go.Scatter(
            x=wavelengths,
            y=result['pyr_component'],
            mode='lines',
            name=pyr_label,
            line=dict(
                color='crimson',
                width=3 if fixed_component == 'PYR' else 2,
                dash='solid' if fixed_component == 'PYR' else 'dash'
            ),
            legendgroup='components',
            showlegend=True
        ),
        row=1, col=1
    )

    # Intercept/Baseline component (if non-zero)
    if result['Intercept'] != 0:
        fig.add_trace(
            go.Scatter(
                x=wavelengths,
                y=result['intercept_component'],
                mode='lines',
                name=f'Baseline ({result["Intercept"]:.4f})',
                line=dict(color='gray', width=1.5, dash='dot'),
                legendgroup='components',
                showlegend=True
            ),
            row=1, col=1
        )

    # --- Residuals Plot ---
    fig.add_trace(
        go.Scatter(
            x=wavelengths,
            y=result['residuals'],
            mode='markers',
            name='Residuals',
            marker=dict(size=3, color='orange', symbol='circle'),
            showlegend=False
        ),
        row=2, col=1
    )

    # Zero line for residuals
    fig.add_trace(
        go.Scatter(
            x=[wavelengths.min(), wavelengths.max()],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )

    # --- Layout Configuration ---

    # Auto-generate title if not provided
    if plot_title is None:
        wavelength_str = f" ({wavelength_range[0]}-{wavelength_range[1]} nm)" if wavelength_range else ""
        fixed_str = ""
        if fixed_component:
            fixed_value = result['NADH_Conc'] if fixed_component == 'NADH' else result['PYR_Conc']
            fixed_str = f" [{fixed_component} Fixed at {fixed_value:.4f} mM]"
        plot_title = f'Spectral Deconvolution{wavelength_str}{fixed_str}'

    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(size=16, color='black'),
            x=0.5,
            xanchor='center'
        ),
        height=700,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        font=dict(size=12)
    )

    # X-axis labels
    fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=1, showgrid=True)
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1, showgrid=True)

    # Y-axis labels
    fig.update_yaxes(title_text="Absorbance", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Residual", row=2, col=1, showgrid=True, zeroline=True)

    return fig