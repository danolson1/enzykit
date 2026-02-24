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
    init_pyr_time=None,
    init_nadh_time=None,
    initial_pyruvate_mM=None,
    method='constrained',
    wavelength_range=(320, 420),
    absorbance_max=2,
    fit_intercept=False,
    plot=False,
    plot_title=None,
    verbose=True
):
    """
    Process PDC (Pyruvate Dehydrogenase Complex) time course data to calculate NADH concentrations.

    This function processes spectrophotometric time course data from a PDC enzyme assay,
    calculating NADH concentrations at each time point using spectral deconvolution.

    **PDC Assay Protocol:**
    The assay is performed in three stages, each corresponding to a key time parameter:
      1. Pyruvate is added to the cuvette. The spectrum at ``init_pyr_time`` is used to
         calculate the initial pyruvate concentration (assuming NADH = 0 at this point).
      2. NADH is added. The spectrum at ``init_nadh_time`` is used to calculate the initial
         NADH concentration (pyruvate fixed at the value from step 1).
      3. The reaction is initiated by adding enzyme (or acetaldehyde). This is
         ``assay_start_time``, which defines t = 0 for the kinetic trace.

    **Pyruvate Concentration Logic:**
    - If init_pyr_time is provided: Calculates pyruvate from that spectrum (assuming NADH=0)
    - If init_pyr_time is None: Must provide initial_pyruvate_mM manually
    - initial_pyruvate_mM is optional when init_pyr_time is given (used only as fallback)

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
        Time point (in seconds) when the assay was initiated (enzyme or acetaldehyde added).
        Defines t = 0 for the kinetic trace (Time_Relative_s).

    init_pyr_time : float, optional (default=None)
        Time point (in seconds) at which pyruvate was added and initial pyruvate
        concentration is measured via spectral deconvolution (assuming NADH = 0).
        If None, must provide initial_pyruvate_mM manually.

    init_nadh_time : float, optional (default=None)
        Time point (in seconds) at which NADH was added and initial NADH concentration
        is measured via spectral deconvolution (pyruvate fixed at the value from
        init_pyr_time).
        If provided, the result DataFrame will contain a non-NaN 'Initial_NADH_mM' column.

    initial_pyruvate_mM : float, optional (default=None)
        Initial pyruvate concentration in mM.
        - If init_pyr_time is provided: Used only as fallback if spectral calculation fails
        - If init_pyr_time is None: REQUIRED - used directly for all calculations

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

    fit_intercept : bool, optional (default=False)
        If True, allows non-zero intercept (baseline offset) in spectral fitting
        If False, forces fit through zero
        Setting to True can help absorb baseline shifts and reduce jaggedness

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
        - 'Blank_Pyruvate_mM': Pyruvate concentration calculated from init_pyr_time spectrum
        - 'Initial_pyruvate_mM': Initial pyruvate concentration (mM)
        - 'Initial_NADH_mM': Initial NADH concentration (mM), or NaN if init_nadh_time not given
        - 'Method': Method used for calculation
        - 'Fixed_Pyruvate': Whether pyruvate was fixed (boolean)

    Examples
    --------
    >>> # Recommended: Automatic pyruvate calculation from spectrum
    >>> results = process_pdc_timecourse(
    ...     spectral_df=cell_1_df,
    ...     standards_df=standards_df,
    ...     assay_start_time=300.0,
    ...     init_pyr_time=60.0,    # Pyruvate added at 60 s
    ...     init_nadh_time=180.0,  # NADH added at 180 s
    ...     plot=True
    ... )

    >>> # Manual pyruvate specification (no spectral measurement)
    >>> results = process_pdc_timecourse(
    ...     spectral_df=cell_1_df,
    ...     standards_df=standards_df,
    ...     assay_start_time=180.0,
    ...     initial_pyruvate_mM=100.0,  # Must specify manually
    ...     plot=True
    ... )

    >>> # Provide fallback in case spectral calculation fails
    >>> results = process_pdc_timecourse(
    ...     spectral_df=cell_1_df,
    ...     standards_df=standards_df,
    ...     assay_start_time=300.0,
    ...     init_pyr_time=60.0,
    ...     initial_pyruvate_mM=100.0,  # Used only if spectral calc fails
    ...     plot=True
    ... )

    Notes
    -----
    - The function assumes a 1 cm path length
    - When init_pyr_time is provided, pyruvate is calculated assuming NADH=0
    - For PDC assays, NADH production indicates pyruvate consumption
    - R² values < 0.95 may indicate poor fit quality
    """

    # Local aliases so internal logic uses the old variable names unchanged
    blank_time = init_pyr_time
    nadh_time = init_nadh_time

    # Coerce numeric parameters that may arrive as strings from pandas object-dtype columns
    # (e.g. when the metadata CSV has '?' values mixed with numbers)
    if blank_time is not None and not pd.isna(blank_time):
        blank_time = float(blank_time)
    if nadh_time is not None and not pd.isna(nadh_time):
        nadh_time = float(nadh_time)
    else:
        nadh_time = None
    if assay_start_time is not None:
        assay_start_time = float(assay_start_time)

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
            print(f"Init pyr time: {blank_time} s (pyruvate will be calculated)")
            if initial_pyruvate_mM is not None:
                print(f"Fallback pyruvate: {initial_pyruvate_mM} mM (if calculation fails)")
        else:
            print(f"Manual pyruvate: {initial_pyruvate_mM} mM (no spectral calculation)")
        if nadh_time is not None:
            print(f"Init NADH time: {nadh_time} s (initial NADH will be calculated)")
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
        blank_pyruvate_mM = _calculate_initial_pyruvate(
            spectral_df, standards_df, spectral_cols, blank_time,
            method, wavelength_range, absorbance_max,
            initial_pyruvate_mM, fit_intercept, verbose
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

    # Calculate initial NADH from nadh_time spectrum (pyruvate fixed at blank_pyruvate_mM)
    initial_nadh_mM = np.nan
    if nadh_time is not None and method != 'single_wavelength':
        initial_nadh_mM = _calculate_initial_nadh(
            spectral_df, standards_df, spectral_cols, nadh_time,
            blank_pyruvate_mM, method, wavelength_range, absorbance_max,
            fit_intercept, verbose
        )

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
            fit_intercept,
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
    results_df['Initial_pyruvate_mM'] = blank_pyruvate_mM if method != 'single_wavelength' else np.nan
    results_df['Initial_NADH_mM'] = initial_nadh_mM

    # Reorder columns for clarity
    col_order = ['Time_s', 'Time_Relative_s', 'NADH_mM', 'Pyruvate_mM',
                 'R_squared', 'Intercept', 'Blank_Pyruvate_mM',
                 'Initial_pyruvate_mM', 'Initial_NADH_mM',
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


def _calculate_initial_pyruvate(
    spectral_df, standards_df, spectral_cols, blank_time,
    method, wavelength_range, absorbance_max,
    fallback_pyruvate_mM, fit_intercept, verbose
):
    """
    Calculate pyruvate concentration at blank_time assuming NADH=0.

    Returns the calculated pyruvate or fallback value if calculation fails.
    """
    if verbose:
        print("Calculating pyruvate concentration at blank time...")
        print(f"  Assumption: NADH = 0 mM at t = {blank_time} s")

    # Check if blank_time is valid
    if pd.isna(blank_time):
        if verbose:
            print(f"  ⚠️  Warning: no blank_time value is present in the metadata.")
            print(f"     Pyruvate concentration was not calculated, and the")
            print(f"     initial_pyruvate_mM value ({fallback_pyruvate_mM:.4f} mM) was used instead.")
        if fallback_pyruvate_mM is not None:
            return fallback_pyruvate_mM
        else:
            raise ValueError(
                "blank_time is NaN and no fallback pyruvate value provided. "
                "Please provide initial_pyruvate_mM."
            )

    # Find the row closest to blank_time
    idx_blank = (spectral_df['Time_s'] - blank_time).abs().idxmin()

    # Check if idx_blank is valid
    if pd.isna(idx_blank):
        if verbose:
            print(f"  ⚠️  Warning: Could not find valid time point near blank_time.")
            print(f"     Pyruvate concentration was not calculated, and the")
            print(f"     initial_pyruvate_mM value ({fallback_pyruvate_mM:.4f} mM) was used instead.")
        if fallback_pyruvate_mM is not None:
            return fallback_pyruvate_mM
        else:
            raise ValueError(
                "Could not find valid blank time point and no fallback provided. "
                "Please provide initial_pyruvate_mM."
            )

    blank_row = spectral_df.loc[idx_blank]
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
            fit_intercept=fit_intercept,
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


def _calculate_initial_nadh(
    spectral_df, standards_df, spectral_cols, nadh_time,
    fixed_pyruvate_mM, method, wavelength_range, absorbance_max,
    fit_intercept, verbose
):
    """
    Calculate initial NADH concentration at nadh_time with pyruvate fixed.

    Returns the calculated NADH concentration, or np.nan if calculation fails.
    """
    if verbose:
        print("Calculating initial NADH concentration at nadh_time...")
        print(f"  Pyruvate fixed at: {fixed_pyruvate_mM:.4f} mM")

    if pd.isna(nadh_time):
        if verbose:
            print("  ⚠️  Warning: nadh_time is NaN. Initial NADH set to NaN.")
        return np.nan

    idx_nadh = (spectral_df['Time_s'] - float(nadh_time)).abs().idxmin()
    if pd.isna(idx_nadh):
        if verbose:
            print("  ⚠️  Warning: Could not find time point near nadh_time. Initial NADH set to NaN.")
        return np.nan

    nadh_row = spectral_df.loc[idx_nadh]
    actual_nadh_time = nadh_row['Time_s']

    spectrum_data = []
    for col in spectral_cols:
        try:
            wavelength = float(col)
            absorbance = nadh_row[col]
            spectrum_data.append({'Wavelength': wavelength, 'Absorbance': absorbance})
        except Exception:
            continue

    spectrum_df_nadh = pd.DataFrame(spectrum_data)

    if method == 'full_spectrum':
        wl_range = None
        abs_max = None
    elif method == 'constrained':
        wl_range = wavelength_range
        abs_max = absorbance_max
    else:
        raise ValueError(f"Unknown method: {method}")

    try:
        result = calculate_concentrations(
            spectrum_df=spectrum_df_nadh,
            standards_df=standards_df,
            wavelength_range=wl_range,
            absorbance_max=abs_max,
            fit_intercept=fit_intercept,
            fixed_pyr=fixed_pyruvate_mM,
            plot=False
        )

        initial_nadh = result['NADH_Conc']
        r_squared = result['R_squared']

        if verbose:
            print(f"  Actual nadh_time: {actual_nadh_time:.1f} s")
            print(f"  Calculated initial NADH: {initial_nadh:.4f} mM")
            print(f"  Fit R²: {r_squared:.6f}")
            if r_squared < 0.95:
                print(f"  ⚠️  Warning: Low R² ({r_squared:.4f}) at nadh_time")
            print()

        return initial_nadh

    except Exception as e:
        if verbose:
            print(f"  ⚠️  Error calculating initial NADH: {e}")
            print("  Initial NADH set to NaN.")
            print()
        return np.nan


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
    fit_intercept,
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
            fit_intercept=fit_intercept,
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
        legend=dict(yanchor="top", y=0.75, xanchor="right", x=0.99)
    )

    fig.show()


def select_modeling_data(timecourse_df, mask_until_rel=None, r2_threshold=0.8, plot=False, plot_title=None):
    """
    Select data points for kinetic modeling from a timecourse DataFrame.

    Filters a timecourse (output of process_pdc_timecourse) to the subset
    suitable for fitting kinetic models:
      - Only points at or after assay start (Time_Relative_s >= 0)
      - Optional exclusion of an initial transition period
      - Exclusion of points with poor spectral fit quality (R² < r2_threshold)

    Parameters
    ----------
    timecourse_df : pandas.DataFrame
        Output of process_pdc_timecourse().

    mask_until_rel : float, optional (default=None)
        Time relative to assay start (s). Points with Time_Relative_s <= this
        value are excluded (transition period at assay start). Pass None to
        skip initial masking.

    r2_threshold : float, optional (default=0.8)
        Points with R_squared < r2_threshold are excluded. Set to 0 to
        disable R² filtering.

    plot : bool, optional (default=False)
        If True, generates a Plotly scatter plot showing masked points (gray)
        and kept points (steelblue), plus a dashed reference line at the
        initial NADH concentration (from Initial_NADH_mM column if available,
        otherwise the first kinetic data point).

    plot_title : str, optional (default=None)
        Title for the plot. Auto-generated if None.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of timecourse_df, reset index, ready for kinetic modeling.
    """
    data = timecourse_df[timecourse_df['Time_Relative_s'] >= 0].copy().reset_index(drop=True)

    # Initial transition mask
    if mask_until_rel is not None:
        initial_mask = data['Time_Relative_s'].values < float(mask_until_rel)
    else:
        initial_mask = np.zeros(len(data), dtype=bool)

    # R² mask
    if 'R_squared' in data.columns and not data['R_squared'].isna().all():
        r2_mask = data['R_squared'].values < r2_threshold
    else:
        r2_mask = np.zeros(len(data), dtype=bool)

    kept = data[~(initial_mask | r2_mask)].copy().reset_index(drop=True)

    # Reference NADH: spectral measurement at init_nadh_time (Initial_NADH_mM column),
    # falling back to the first kinetic data point if the column is absent or NaN.
    if 'Initial_NADH_mM' in data.columns and not pd.isna(data['Initial_NADH_mM'].iloc[0]):
        ref_nadh = float(data['Initial_NADH_mM'].iloc[0])
    else:
        ref_nadh = float(data['NADH_mM'].iloc[0])

    # Plateau trimming: if the tail of kept has NADH < 5% of ref_nadh,
    # limit those plateau points to 10% of the total kept count.
    if not kept.empty and ref_nadh > 0:
        plateau_threshold = 0.05 * ref_nadh
        above = kept['NADH_mM'].values >= plateau_threshold
        if not above.all() and above.any():
            last_active_idx = int(above.nonzero()[0][-1])
            n_plateau = len(kept) - (last_active_idx + 1)
            n_max_plateau = max(1, round(0.10 * len(kept)))
            if n_plateau > n_max_plateau:
                kept = kept.iloc[:last_active_idx + 1 + n_max_plateau].copy().reset_index(drop=True)

    if plot and len(data) > 0:
        kept_times = set(kept['Time_s'].values)
        masked_data = data[~data['Time_s'].isin(kept_times)]
        keep_data = data[data['Time_s'].isin(kept_times)]
        has_r2 = 'R_squared' in data.columns and not data['R_squared'].isna().all()

        hover_suffix = 'R²: %{customdata[1]:.3f}<extra></extra>' if has_r2 else '<extra></extra>'

        fig = go.Figure()

        if len(masked_data) > 0:
            fig.add_trace(go.Scatter(
                x=masked_data['Time_Relative_s'],
                y=masked_data['NADH_mM'],
                mode='markers',
                marker=dict(color='lightgray', size=6),
                name='Masked',
                customdata=masked_data[['Time_s', 'R_squared']].values if has_r2 else masked_data[['Time_s']].values,
                hovertemplate=(
                    'Relative time: %{x:.1f} s<br>'
                    'Original time: %{customdata[0]:.1f} s<br>'
                    'NADH: %{y:.4f} mM<br>'
                    + hover_suffix
                ),
            ))

        fig.add_trace(go.Scatter(
            x=keep_data['Time_Relative_s'],
            y=keep_data['NADH_mM'],
            mode='markers',
            marker=dict(color='steelblue', size=6),
            name='Used for modeling',
            customdata=keep_data[['Time_s', 'R_squared']].values if has_r2 else keep_data[['Time_s']].values,
            hovertemplate=(
                'Relative time: %{x:.1f} s<br>'
                'Original time: %{customdata[0]:.1f} s<br>'
                'NADH: %{y:.4f} mM<br>'
                + hover_suffix
            ),
        ))

        fig.add_hline(
            y=ref_nadh,
            line=dict(color='red', dash='dash', width=1.5),
            annotation_text=f'Initial NADH = {ref_nadh:.3f} mM',
            annotation_position='top right',
        )

        n_masked = len(data) - len(kept)
        fig.update_layout(
            title=plot_title or 'Modeling Data Selection',
            xaxis_title='Time (s)',
            yaxis_title='NADH (mM)',
            template='plotly_white',
        )
        fig.show()
        print(f"Total points after assay start : {len(data)}")
        print(f"  Used for modeling            : {len(kept)}")
        print(f"  Masked (transition + R²<{r2_threshold}) : {n_masked}")

    return kept


def fit_nadh_degradation(timecourse_df, r2_threshold=0.8, plot=False, plot_title=None):
    """
    Fit an exponential decay model to NADH concentration data.

    Fits NADH(t) = NADH0 * exp(-k * t) where t is time relative to the first
    data point in the supplied DataFrame (so NADH0 represents NADH at the start
    of the decay window, not necessarily the assay start).

    Typically called with a pre-windowed DataFrame — e.g. the NADH-degradation
    standard period, or the pre-assay holding period from process_pdc_timecourse.

    Parameters
    ----------
    timecourse_df : pandas.DataFrame
        DataFrame containing at minimum 'Time_Relative_s' and 'NADH_mM' columns.
        Typically a slice of process_pdc_timecourse() output covering the decay
        window of interest.

    r2_threshold : float, optional (default=0.8)
        Points with R_squared < r2_threshold are excluded before fitting.
        Set to 0 to use all points.

    plot : bool, optional (default=False)
        If True, generates a Plotly figure showing data and fit curve.

    plot_title : str, optional
        Title for the plot. Auto-generated if None.

    Returns
    -------
    dict
        - 'k_s'         : float — decay constant (s⁻¹)
        - 'NADH0_fit'   : float — fitted NADH at start of decay window (mM)
        - 'half_life_h' : float — half-life in hours
        - 'n_points'    : int   — number of points used in the fit
        - 'fig'         : plotly.graph_objects.Figure or None

    Raises
    ------
    ValueError
        If fewer than 3 data points remain after R² filtering.
    """
    from scipy.optimize import curve_fit

    data = timecourse_df.copy()

    # Apply R² filter if column is present
    if 'R_squared' in data.columns and not data['R_squared'].isna().all():
        data = data[data['R_squared'] >= r2_threshold].copy()

    if len(data) < 3:
        raise ValueError(
            f"Not enough data points ({len(data)}) for exponential fit "
            f"after R² filtering (threshold={r2_threshold})."
        )

    # Shift time so t=0 is the first point in the window
    t_offset = data['Time_Relative_s'].values.min()
    t_fit = data['Time_Relative_s'].values - t_offset
    nadh = data['NADH_mM'].values

    def _exp_decay(t, nadh0, k):
        return nadh0 * np.exp(-k * t)

    popt, _ = curve_fit(
        _exp_decay, t_fit, nadh,
        p0=[nadh.max(), 1e-4],
        bounds=(0, np.inf),
    )
    nadh0_fit, k_fit = popt
    half_life_h = np.log(2) / k_fit / 3600

    result = {
        'k_s': k_fit,
        'NADH0_fit': nadh0_fit,
        'half_life_h': half_life_h,
        'n_points': len(data),
        'fig': None,
    }

    if plot:
        t_abs = data['Time_Relative_s'].values
        t_fine = np.linspace(t_fit.min(), t_fit.max(), 300)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_abs,
            y=nadh,
            mode='markers',
            marker=dict(size=6, color='steelblue'),
            name='Data',
        ))
        fig.add_trace(go.Scatter(
            x=t_fine + t_offset,
            y=_exp_decay(t_fine, *popt),
            mode='lines',
            line=dict(color='red'),
            name=f'Exp decay (k = {k_fit:.2e} s⁻¹, t½ = {half_life_h:.1f} h)',
        ))
        fig.update_layout(
            title=plot_title or 'NADH Exponential Decay Fit',
            xaxis_title='Time Relative to Assay Start (s)',
            yaxis_title='NADH (mM)',
            template='plotly_white',
        )
        fig.show()
        result['fig'] = fig

    return result


def calculate_max_slope(
    timecourse_data,
    window_fraction=0.2,
    min_window_size=5,
    min_r_squared=0.9,
    slope_direction='negative',
    plot=False,
    plot_title=None
):
    """
    Calculate maximum slope (velocity) from timecourse data using rolling linear regression.

    This function finds the region of maximum velocity by performing linear regression
    over a rolling window of the timecourse data. Useful for determining the initial
    velocity (V) from enzyme assay data.

    Parameters
    ----------
    timecourse_data : pandas.DataFrame
        DataFrame from process_pdc_timecourse() containing 'Time_Relative_s' and 'NADH_mM' columns

    window_fraction : float, optional (default=0.2)
        Fraction of total assay duration to use as window size (0.2 = 1/5 of duration)

    min_window_size : int, optional (default=5)
        Minimum number of data points required in the regression window

    min_r_squared : float, optional (default=0.9)
        Minimum R² value required for the fit to be considered valid

    slope_direction : str, optional (default='negative')
        Direction to maximize: 'negative' (most negative slope, typical for NADH consumption)
        or 'positive' (most positive slope, for NADH production)

    plot : bool, optional (default=False)
        If True, generates a plot showing NADH vs time with the maximum slope region highlighted

    plot_title : str, optional
        Custom title for the plot. If None, auto-generates a title.

    Returns
    -------
    dict
        Dictionary containing:
        - 'V_max_slope': Maximum slope in µM/min (or None if insufficient data or R² too low)
        - 'slope_start_time': Start time of best slope window in seconds (or None)
        - 'slope_end_time': End time of best slope window in seconds (or None)
        - 'slope_r_squared': R² value of the best fit (or None)
        - 'slope_intercept': Y-intercept of the best fit line (or None)
        - 'window_size': Number of points used in the window (or None)
        - 'fig': Plotly figure object (or None if plot=False)

    Notes
    -----
    - Data is automatically filtered to Time_Relative_s >= 0 before analysis
    - Returns dict with None values if:
        * Fewer than min_window_size data points after filtering
        * Best R² is below min_r_squared threshold
    - For 'negative' direction, finds most negative slope (minimum)
    - For 'positive' direction, finds most positive slope (maximum)

    Examples
    --------
    >>> results = process_pdc_timecourse(...)
    >>> slope_data = calculate_max_slope(results, plot=True)
    >>> print(f"Maximum velocity: {slope_data['V_max_slope']:.2f} µM/min")
    >>> print(f"R²: {slope_data['slope_r_squared']:.3f}")
    """
    from scipy import stats

    # Filter to only data after assay start (Time_Relative_s >= 0)
    data = timecourse_data[timecourse_data['Time_Relative_s'] >= 0].copy()

    # Check if sufficient data
    if len(data) < min_window_size:
        return {
            'V_max_slope': None,
            'slope_start_time': None,
            'slope_end_time': None,
            'slope_r_squared': None,
            'slope_intercept': None,
            'window_size': None,
            'fig': None
        }

    # Calculate window size: fraction of assay duration, minimum min_window_size points
    window_size = max(min_window_size, int(len(data) * window_fraction))

    # Ensure window_size doesn't exceed data length
    window_size = min(window_size, len(data))

    # Calculate rolling linear regression
    if slope_direction == 'negative':
        best_slope = np.inf  # Start with infinity, look for most negative
    else:  # 'positive'
        best_slope = -np.inf  # Start with -infinity, look for most positive

    best_window_start = None
    best_window_end = None
    best_r_squared = None
    best_intercept = None

    for i in range(len(data) - window_size + 1):
        window_data = data.iloc[i:i+window_size]

        # Perform linear regression: NADH = slope * Time + intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            window_data['Time_Relative_s'],
            window_data['NADH_mM']
        )

        # Track best slope based on direction
        if slope_direction == 'negative':
            # For negative direction, we want the most negative slope (minimum)
            if slope < best_slope:
                best_slope = slope
                best_window_start = window_data['Time_Relative_s'].iloc[0]
                best_window_end = window_data['Time_Relative_s'].iloc[-1]
                best_r_squared = r_value ** 2
                best_intercept = intercept
        else:  # 'positive'
            # For positive direction, we want the most positive slope (maximum)
            if slope > best_slope:
                best_slope = slope
                best_window_start = window_data['Time_Relative_s'].iloc[0]
                best_window_end = window_data['Time_Relative_s'].iloc[-1]
                best_r_squared = r_value ** 2
                best_intercept = intercept

    # Check if R² meets threshold
    if best_r_squared is None or best_r_squared < min_r_squared:
        # R² too low, return None values
        result = {
            'V_max_slope': None,
            'slope_start_time': None,
            'slope_end_time': None,
            'slope_r_squared': best_r_squared,  # Include actual R² even if too low
            'slope_intercept': None,
            'window_size': None,
            'fig': None
        }
    else:
        # Valid result - convert slope from mM/s to µM/min (multiply by 1000 × 60 = 60000)
        result = {
            'V_max_slope': best_slope * 60000,
            'slope_start_time': best_window_start,
            'slope_end_time': best_window_end,
            'slope_r_squared': best_r_squared,
            'slope_intercept': best_intercept,
            'window_size': window_size,
            'fig': None
        }

    # Generate plot if requested
    if plot and result['V_max_slope'] is not None:
        fig = go.Figure()

        # Plot all NADH data
        fig.add_trace(
            go.Scatter(
                x=data['Time_Relative_s'],
                y=data['NADH_mM'],
                mode='markers+lines',
                name='NADH',
                marker=dict(size=4, color='blue'),
                line=dict(color='blue', width=1)
            )
        )

        # Plot best slope line
        time_fit = np.array([best_window_start, best_window_end])
        nadh_fit = best_slope * time_fit + best_intercept

        fig.add_trace(
            go.Scatter(
                x=time_fit,
                y=nadh_fit,
                mode='lines',
                name='Max Slope',
                line=dict(color='red', width=3, dash='dash')
            )
        )

        # Add annotation - offset to the right by 20% of window duration
        time_offset = (best_window_end - best_window_start) * 0.2
        annotation_x = best_window_start + time_offset
        annotation_y = best_slope * annotation_x + best_intercept

        # Convert slope to µM/min for display (mM/s × 60000 = µM/min)
        slope_uM_per_min = best_slope * 60000

        fig.add_annotation(
            x=annotation_x,
            y=annotation_y,
            text=f"V = {slope_uM_per_min:.2f} µM/min<br>R² = {best_r_squared:.3f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red',
            ax=-60,
            ay=-40,
            bgcolor='white',
            bordercolor='red',
            borderwidth=2
        )

        # Update layout
        if plot_title is None:
            plot_title = f'Maximum Slope Analysis (V = {slope_uM_per_min:.2f} µM/min)'

        # Calculate y-axis range with minimum of -0.1
        y_min = max(-0.1, data['NADH_mM'].min() * 0.95)
        y_max = data['NADH_mM'].max() * 1.05

        fig.update_layout(
            title=plot_title,
            xaxis_title='Time from Assay Start (s)',
            yaxis_title='NADH Concentration (mM)',
            template='plotly_white',
            height=500,
            hovermode='x unified',
            yaxis=dict(range=[y_min, y_max])
        )

        result['fig'] = fig
        fig.show()

    return result


# =============================================================================
# FUTURE: Additional Enzyme Assay Functions
# =============================================================================

# TODO: Implement process_fdh_timecourse()
# Function for processing Formate Dehydrogenase (FDH) time course data
# Will follow similar pattern to process_pdc_timecourse() but adapted for FDH kinetics

# TODO: Implement process_adh_timecourse()
# Function for processing Alcohol Dehydrogenase (ADH) time course data
# Will follow similar pattern to process_pdc_timecourse() but adapted for ADH kinetics
