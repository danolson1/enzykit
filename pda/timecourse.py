"""
Time course processing functions for enzyme assays.

This module provides functions for processing time course data from various enzyme assays
including PDC (Pyruvate Dehydrogenase Complex), FDH (Formate Dehydrogenase), 
and ADH (Alcohol Dehydrogenase).
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Import from other modules in the package
try:
    from .spectral import calculate_concentrations
except ImportError:
    # Fallback for when running as standalone script
    from spectral import calculate_concentrations


def process_pdc_timecourse(
    spectral_df,
    standards_df,
    assay_start_time,
    blank_time=None,
    initial_pyruvate_mM=None,
    method='constrained',
    wavelength_range=(320, 420),
    absorbance_max=2,
    plot=False,
    plot_title=None,
    verbose=True
):
    """
    Process PDC (Pyruvate Dehydrogenase Complex) time course data to calculate NADH concentrations.

    This function processes spectrophotometric time course data from a PDC enzyme assay,
    calculating NADH concentrations at each time point using spectral deconvolution.

    **Pyruvate Concentration Logic:**
    - If blank_time is provided: Calculates pyruvate from blank spectrum (assuming NADH=0)
    - If blank_time is None: Must provide initial_pyruvate_mM manually
    - initial_pyruvate_mM is optional when blank_time is given (used only as fallback)

    Parameters
    ----------
    spectral_df : pandas.DataFrame
        DataFrame containing spectral time course data with:
        - 'Time_s' column: time in seconds
        - Wavelength columns: named with wavelength values (e.g., '340', '350', etc.)
        - Optional 'sample' column for identification

    standards_df : pandas.DataFrame
        DataFrame with extinction coefficient standards containing:
        - 'Wavelength' column: wavelength values
        - 'NADH_Coeff' column: NADH extinction coefficients (mM⁻¹cm⁻¹)
        - 'PYR_Coeff' column: Pyruvate extinction coefficients (mM⁻¹cm⁻¹)

    assay_start_time : float
        Time point (in seconds) when the assay was initiated (e.g., when substrate was added)

    blank_time : float, optional (default=None)
        Time point (in seconds) to use for blank/baseline correction
        If provided, pyruvate concentration is calculated from this spectrum (assuming NADH=0)
        If None, must provide initial_pyruvate_mM manually

    initial_pyruvate_mM : float, optional (default=None)
        Initial pyruvate concentration in mM
        - If blank_time is provided: Used only as fallback if calculation fails
        - If blank_time is None: REQUIRED - used directly for all calculations

    method : str, optional (default='constrained')
        Method for NADH calculation:
        - 'single_wavelength': Use Abs340 only with blank subtraction
        - 'full_spectrum': Use entire spectrum for deconvolution
        - 'constrained': Use wavelength_range and absorbance_max constraints (recommended)

    wavelength_range : tuple, optional (default=(320, 420))
        Tuple of (min_wavelength, max_wavelength) in nm
        Only used if method='constrained'

    absorbance_max : float, optional (default=2.5)
        Maximum absorbance threshold to exclude saturated data
        Only used if method='constrained'

    plot : bool, optional (default=False)
        If True, generates diagnostic plots

    plot_title : str, optional (default=None)
        Custom title for plots. Auto-generates if None.

    verbose : bool, optional (default=True)
        If True, prints progress and summary statistics

    Returns
    -------
    pandas.DataFrame
        DataFrame with the following columns:
        - 'Time_s': Time in seconds
        - 'Time_Relative_s': Time relative to assay_start_time (0 = assay start)
        - 'NADH_mM': NADH concentration in mM
        - 'Pyruvate_mM': Pyruvate concentration in mM (fixed value)
        - 'R_squared': Goodness of fit for each time point
        - 'Intercept': Baseline offset at each time point
        - 'Blank_Pyruvate_mM': Pyruvate concentration calculated from blank
        - 'Method': Method used for calculation
        - 'Fixed_Pyruvate': Whether pyruvate was fixed (boolean)

    Examples
    --------
    >>> # Recommended: Automatic pyruvate calculation from blank
    >>> results = process_pdc_timecourse(
    ...     spectral_df=cell_1_df,
    ...     standards_df=standards_df,
    ...     assay_start_time=180.0,
    ...     blank_time=180.0,  # Pyruvate auto-calculated from this spectrum
    ...     plot=True
    ... )

    >>> # Manual pyruvate specification (no blank_time)
    >>> results = process_pdc_timecourse(
    ...     spectral_df=cell_1_df,
    ...     standards_df=standards_df,
    ...     assay_start_time=180.0,
    ...     initial_pyruvate_mM=100.0,  # Must specify manually
    ...     plot=True
    ... )

    >>> # Provide fallback in case blank calculation fails
    >>> results = process_pdc_timecourse(
    ...     spectral_df=cell_1_df,
    ...     standards_df=standards_df,
    ...     assay_start_time=180.0,
    ...     blank_time=180.0,
    ...     initial_pyruvate_mM=100.0,  # Used only if blank calc fails
    ...     plot=True
    ... )

    Notes
    -----
    - The function assumes a 1 cm path length
    - When blank_time is provided, pyruvate is calculated assuming NADH=0
    - For PDC assays, NADH production indicates pyruvate consumption
    - R² values < 0.95 may indicate poor fit quality
    """

    # Validate parameters
    if blank_time is None and initial_pyruvate_mM is None:
        import warnings
        warnings.warn(
            "Neither blank_time nor initial_pyruvate_mM was specified. "
            "Setting initial_pyruvate_mM to 0. This assumes the spectrophotometer "
            "was blanked after pyruvate addition.",
            UserWarning
        )
        initial_pyruvate_mM = 0.0

    if verbose:
        print("="*80)
        print("PDC TIME COURSE PROCESSING")
        print("="*80)
        print(f"Method: {method}")
        print(f"Assay start time: {assay_start_time} s")
        if blank_time is not None:
            print(f"Blank time: {blank_time} s (pyruvate will be calculated)")
            if initial_pyruvate_mM is not None:
                print(f"Fallback pyruvate: {initial_pyruvate_mM} mM (if calculation fails)")
        else:
            print(f"Manual pyruvate: {initial_pyruvate_mM} mM (no blank calculation)")
        if method == 'constrained':
            print(f"Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} nm")
            print(f"Max absorbance: {absorbance_max}")
        print()

    # Identify wavelength columns
    non_spectral_cols = ['sample', 'Time_s', 'filename', 'NADH_Conc_SingleWav',
                         'NADH_Conc_Spectral', 'NADH_Method1', 'NADH_Method2',
                         'NADH_Method3', 'NADH_Method4']
    spectral_cols = [c for c in spectral_df.columns if c not in non_spectral_cols]

    # Determine pyruvate concentration to use
    if blank_time is not None and method != 'single_wavelength':
        # Calculate pyruvate from blank spectrum
        blank_pyruvate_mM = _calculate_blank_pyruvate(
            spectral_df, standards_df, spectral_cols, blank_time,
            method, wavelength_range, absorbance_max,
            initial_pyruvate_mM, verbose
        )

        # If both blank_time and initial_pyruvate_mM were specified, compare them
        if initial_pyruvate_mM is not None:
            percent_difference = abs(blank_pyruvate_mM - initial_pyruvate_mM) / initial_pyruvate_mM * 100

            # Check if measured value is within 50% of specified value
            if percent_difference > 50:
                import warnings
                warnings.warn(
                    f"Measured pyruvate concentration is out of range!\n"
                    f"  Specified value: {initial_pyruvate_mM:.4f} mM\n"
                    f"  Measured value:  {blank_pyruvate_mM:.4f} mM\n"
                    f"  Percent difference: {percent_difference:.1f}%\n"
                    f"The measured value differs by more than 50% from the specified value.",
                    UserWarning
                )
    elif initial_pyruvate_mM is not None:
        # Use manually specified pyruvate
        blank_pyruvate_mM = initial_pyruvate_mM
        if verbose and method != 'single_wavelength':
            print(f"Using manually specified pyruvate: {blank_pyruvate_mM:.4f} mM")
            print()
    else:
        raise ValueError("This should not happen - parameter validation failed")

    # Process based on method
    if method == 'single_wavelength':
        if blank_time is None:
            raise ValueError("single_wavelength method requires blank_time to be specified")
        results_df = _process_single_wavelength(
            spectral_df, blank_time, verbose
        )
        results_df['Blank_Pyruvate_mM'] = np.nan  # Not applicable
    else:
        results_df = _process_spectral_deconvolution(
            spectral_df, standards_df, spectral_cols,
            method, wavelength_range, absorbance_max,
            blank_pyruvate_mM,
            verbose
        )
        results_df['Blank_Pyruvate_mM'] = blank_pyruvate_mM

    # Add relative time column (0 = assay start)
    results_df['Time_Relative_s'] = results_df['Time_s'] - assay_start_time

    # Add metadata columns
    results_df['Method'] = method
    results_df['Fixed_Pyruvate'] = True if method != 'single_wavelength' else False
    results_df['Assay_Start_s'] = assay_start_time
    results_df['Blank_Time_s'] = blank_time if blank_time is not None else np.nan

    # Reorder columns for clarity
    col_order = ['Time_s', 'Time_Relative_s', 'NADH_mM', 'Pyruvate_mM',
                 'R_squared', 'Intercept', 'Blank_Pyruvate_mM',
                 'Method', 'Fixed_Pyruvate', 'Assay_Start_s', 'Blank_Time_s']
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    # Print summary statistics
    if verbose:
        _print_summary_statistics(results_df, assay_start_time)

    # Generate plots if requested
    if plot:
        _generate_diagnostic_plots(
            results_df, spectral_df, spectral_cols, standards_df,
            assay_start_time, plot_title, method
        )

    return results_df


def _calculate_blank_pyruvate(
    spectral_df, standards_df, spectral_cols, blank_time,
    method, wavelength_range, absorbance_max,
    fallback_pyruvate_mM, verbose
):
    """
    Calculate pyruvate concentration at blank_time assuming NADH=0.

    Returns the calculated pyruvate or fallback value if calculation fails.
    """
    if verbose:
        print("Calculating pyruvate concentration at blank time...")
        print(f"  Assumption: NADH = 0 mM at t = {blank_time} s")

    # Find the row closest to blank_time
    idx_blank = (spectral_df['Time_s'] - blank_time).abs().idxmin()
    blank_row = spectral_df.iloc[idx_blank]
    actual_blank_time = blank_row['Time_s']

    # Build spectrum dataframe for blank time point
    spectrum_data = []
    for col in spectral_cols:
        try:
            wavelength = float(col)
            absorbance = blank_row[col]
            spectrum_data.append({'Wavelength': wavelength, 'Absorbance': absorbance})
        except:
            continue

    spectrum_df_blank = pd.DataFrame(spectrum_data)

    # Set parameters based on method
    if method == 'full_spectrum':
        wl_range = None
        abs_max = None
    elif method == 'constrained':
        wl_range = wavelength_range
        abs_max = absorbance_max
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate concentrations with NADH fixed at 0
    try:
        result = calculate_concentrations(
            spectrum_df=spectrum_df_blank,
            standards_df=standards_df,
            wavelength_range=wl_range,
            absorbance_max=abs_max,
            fit_intercept=True,
            fixed_nadh=0.0,  # Fix NADH at 0 for blank
            plot=False
        )

        blank_pyruvate = result['PYR_Conc']
        r_squared = result['R_squared']

        if verbose:
            print(f"  Actual blank time: {actual_blank_time:.1f} s")
            print(f"  Calculated pyruvate: {blank_pyruvate:.4f} mM")
            print(f"  Fit R²: {r_squared:.6f}")

            # Show difference from fallback if provided
            if fallback_pyruvate_mM is not None:
                diff = blank_pyruvate - fallback_pyruvate_mM
                pct_diff = 100 * diff / fallback_pyruvate_mM
                print(f"  Difference from fallback: {diff:+.4f} mM ({pct_diff:+.2f}%)")

            if r_squared < 0.95:
                print(f"  ⚠️  Warning: Low R² ({r_squared:.4f}) at blank time")

            print()

        return blank_pyruvate

    except Exception as e:
        if verbose:
            print(f"  ⚠️  Error calculating blank pyruvate: {e}")
            if fallback_pyruvate_mM is not None:
                print(f"  Using fallback value: {fallback_pyruvate_mM:.4f} mM")
            else:
                print(f"  ERROR: No fallback value provided!")
            print()

        if fallback_pyruvate_mM is not None:
            return fallback_pyruvate_mM
        else:
            raise ValueError(
                "Failed to calculate pyruvate from blank and no fallback value provided. "
                "Please provide initial_pyruvate_mM as a fallback."
            )


def _process_single_wavelength(spectral_df, blank_time, verbose):
    """Process using single wavelength (340 nm) method"""
    col_340 = '340'

    if col_340 not in spectral_df.columns:
        raise ValueError(f"Column '{col_340}' not found in spectral_df")

    # Get blank value
    idx_blank = (spectral_df['Time_s'] - blank_time).abs().idxmin()
    blank_340 = spectral_df.loc[idx_blank, col_340]

    # Calculate NADH
    epsilon_340 = 6.22  # mM^-1 cm^-1
    nadh_conc = (spectral_df[col_340] - blank_340) / epsilon_340

    if verbose:
        print(f"Single wavelength method:")
        print(f"  Blank Abs340: {blank_340:.6f}")
        print(f"  Extinction coeff: {epsilon_340} mM⁻¹cm⁻¹")
        print()

    # Create results dataframe
    results_df = pd.DataFrame({
        'Time_s': spectral_df['Time_s'],
        'NADH_mM': nadh_conc,
        'Pyruvate_mM': np.nan,
        'R_squared': np.nan,
        'Intercept': -blank_340
    })

    return results_df


def _process_spectral_deconvolution(
    spectral_df, standards_df, spectral_cols,
    method, wavelength_range, absorbance_max,
    fixed_pyruvate_mM,
    verbose
):
    """Process using spectral deconvolution methods with fixed pyruvate"""

    nadh_concentrations = []
    pyr_concentrations = []
    r_squared_values = []
    intercepts = []

    total_points = len(spectral_df)

    if verbose:
        print(f"Processing {total_points} time points with fixed pyruvate = {fixed_pyruvate_mM:.4f} mM...")

    for idx, row in spectral_df.iterrows():
        # Build spectrum dataframe
        spectrum_data = []
        for col in spectral_cols:
            try:
                wavelength = float(col)
                absorbance = row[col]
                spectrum_data.append({'Wavelength': wavelength, 'Absorbance': absorbance})
            except:
                continue

        spectrum_df_temp = pd.DataFrame(spectrum_data)

        # Set parameters based on method
        if method == 'full_spectrum':
            wl_range = None
            abs_max = None
        elif method == 'constrained':
            wl_range = wavelength_range
            abs_max = absorbance_max
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate concentrations with FIXED PYRUVATE
        result = calculate_concentrations(
            spectrum_df=spectrum_df_temp,
            standards_df=standards_df,
            wavelength_range=wl_range,
            absorbance_max=abs_max,
            fit_intercept=True,
            fixed_pyr=fixed_pyruvate_mM,
            plot=False
        )

        nadh_concentrations.append(result['NADH_Conc'])
        pyr_concentrations.append(result['PYR_Conc'])
        r_squared_values.append(result['R_squared'])
        intercepts.append(result['Intercept'])

        # Progress indicator
        if verbose and (idx % 20 == 0 or idx == total_points - 1):
            print(f"  Processing: {idx+1}/{total_points} time points...", end='\r')

    if verbose:
        print()
        print()

    # Create results dataframe
    results_df = pd.DataFrame({
        'Time_s': spectral_df['Time_s'].values,
        'NADH_mM': nadh_concentrations,
        'Pyruvate_mM': pyr_concentrations,
        'R_squared': r_squared_values,
        'Intercept': intercepts
    })

    return results_df


def _print_summary_statistics(results_df, assay_start_time):
    """Print summary statistics for the processed data"""
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Blank pyruvate info
    if 'Blank_Pyruvate_mM' in results_df.columns and not pd.isna(results_df['Blank_Pyruvate_mM'].iloc[0]):
        print(f"\nFixed Pyruvate Concentration: {results_df['Blank_Pyruvate_mM'].iloc[0]:.4f} mM")
        print("  (Calculated from blank spectrum assuming NADH=0)")

    # Overall statistics
    print("\nOverall NADH Concentration:")
    print(f"  Mean: {results_df['NADH_mM'].mean():.6f} mM")
    print(f"  Std:  {results_df['NADH_mM'].std():.6f} mM")
    print(f"  Min:  {results_df['NADH_mM'].min():.6f} mM")
    print(f"  Max:  {results_df['NADH_mM'].max():.6f} mM")

    if 'R_squared' in results_df.columns and not results_df['R_squared'].isna().all():
        print("\nFit Quality (R²):")
        print(f"  Mean: {results_df['R_squared'].mean():.6f}")
        print(f"  Min:  {results_df['R_squared'].min():.6f}")
        poor_fits = (results_df['R_squared'] < 0.95).sum()
        if poor_fits > 0:
            print(f"  ⚠️  Warning: {poor_fits} time points with R² < 0.95")

    # Before/after assay start
    before_start = results_df[results_df['Time_s'] < assay_start_time]
    after_start = results_df[results_df['Time_s'] >= assay_start_time]

    if len(before_start) > 0:
        print(f"\nBefore Assay Start (t < {assay_start_time} s):")
        print(f"  NADH: {before_start['NADH_mM'].mean():.6f} ± {before_start['NADH_mM'].std():.6f} mM")

    if len(after_start) > 0:
        print(f"\nAfter Assay Start (t ≥ {assay_start_time} s):")
        print(f"  NADH: {after_start['NADH_mM'].mean():.6f} ± {after_start['NADH_mM'].std():.6f} mM")

        # Calculate rate if there are enough points
        if len(after_start) > 5:
            initial_phase = after_start[after_start['Time_Relative_s'] <= 30].head(10)
            if len(initial_phase) > 2:
                coeffs = np.polyfit(initial_phase['Time_Relative_s'],
                                   initial_phase['NADH_mM'], 1)
                initial_rate = coeffs[0]
                print(f"  Initial rate: {initial_rate:.6f} mM/s ({initial_rate*60:.6f} mM/min)")

    print()


def _generate_diagnostic_plots(
    results_df, spectral_df, spectral_cols, standards_df,
    assay_start_time, plot_title, method
):
    """Generate diagnostic plots for the time course"""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'NADH Concentration vs Time',
            'Fit Quality (R²) vs Time',
            'Intercept vs Time',
            'Sample Spectra at Key Time Points'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Panel 1: NADH concentration
    fig.add_trace(
        go.Scatter(
            x=results_df['Time_Relative_s'],
            y=results_df['NADH_mM'],
            mode='lines+markers',
            name='NADH',
            line=dict(color='royalblue', width=2),
            marker=dict(size=5)
        ),
        row=1, col=1
    )

    fig.add_vline(x=0, line_dash="dash", line_color="red",
                  annotation_text="Assay Start", row=1, col=1)

    # Panel 2: R² values
    if 'R_squared' in results_df.columns and not results_df['R_squared'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=results_df['Time_Relative_s'],
                y=results_df['R_squared'],
                mode='lines+markers',
                name='R²',
                line=dict(color='green', width=2),
                marker=dict(size=5),
                showlegend=False
            ),
            row=1, col=2
        )

        fig.add_hline(y=0.95, line_dash="dash", line_color="gray",
                     annotation_text="R²=0.95", row=1, col=2)

    # Panel 3: Intercept
    if 'Intercept' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Time_Relative_s'],
                y=results_df['Intercept'],
                mode='lines+markers',
                name='Intercept',
                line=dict(color='gray', width=2),
                marker=dict(size=5),
                showlegend=False
            ),
            row=2, col=1
        )

    # Panel 4: Sample spectra
    n_spectra = min(4, len(spectral_df))
    indices = np.linspace(0, len(spectral_df)-1, n_spectra, dtype=int)
    colors = ['blue', 'green', 'orange', 'red']

    for i, idx in enumerate(indices):
        row = spectral_df.iloc[idx]
        wavelengths = []
        absorbances = []

        for col in spectral_cols:
            try:
                wl = float(col)
                if 250 <= wl <= 500:
                    wavelengths.append(wl)
                    absorbances.append(row[col])
            except:
                continue

        time_point = row['Time_s']
        relative_time = time_point - assay_start_time

        fig.add_trace(
            go.Scatter(
                x=wavelengths,
                y=absorbances,
                mode='lines',
                name=f't={relative_time:.0f}s',
                line=dict(color=colors[i], width=2)
            ),
            row=2, col=2
        )

    # Update axes
    fig.update_xaxes(title_text="Time from Assay Start (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time from Assay Start (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time from Assay Start (s)", row=2, col=1)
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=2)

    fig.update_yaxes(title_text="[NADH] (mM)", row=1, col=1)
    fig.update_yaxes(title_text="R²", row=1, col=2)
    fig.update_yaxes(title_text="Intercept", row=2, col=1)
    fig.update_yaxes(title_text="Absorbance", row=2, col=2)

    # Layout
    if plot_title is None:
        blank_pyr = results_df['Blank_Pyruvate_mM'].iloc[0] if 'Blank_Pyruvate_mM' in results_df.columns else None
        if blank_pyr is not None and not pd.isna(blank_pyr):
            plot_title = f"PDC Time Course Analysis ({method} method, PYR={blank_pyr:.2f}mM)"
        else:
            plot_title = f"PDC Time Course Analysis ({method} method)"

    fig.update_layout(
        height=800,
        title_text=plot_title,
        template='plotly_white',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    fig.show()


# =============================================================================
# FUTURE: Additional Enzyme Assay Functions
# =============================================================================

# TODO: Implement process_fdh_timecourse()
# Function for processing Formate Dehydrogenase (FDH) time course data
# Will follow similar pattern to process_pdc_timecourse() but adapted for FDH kinetics

# TODO: Implement process_adh_timecourse()
# Function for processing Alcohol Dehydrogenase (ADH) time course data
# Will follow similar pattern to process_pdc_timecourse() but adapted for ADH kinetics
