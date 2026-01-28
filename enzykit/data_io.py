"""
Data I/O utilities for PDA spectrophotometer analysis.

This module provides functions for loading and parsing data from photodiode array
spectrophotometers, including standard spectra (.WAV files) and kinetic data (.CSV files).
"""

import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def parse_wav_files(wav_file_paths: List[str]) -> pd.DataFrame:
    """
    Parse .WAV files from Agilent spectrophotometer containing standard spectra.
    
    This function reads multiple .WAV files, extracts wavelength and absorbance data,
    and combines them into a single DataFrame. It also attempts to extract compound
    names and concentrations from filenames.
    
    Parameters
    ----------
    wav_file_paths : List[str]
        List of file paths to .WAV files
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'Wavelength': Wavelength in nm
        - 'Absorbance': Absorbance value
        - 'Filename': Original filename
        - 'Compound': Compound name (e.g., 'NADH', 'PYR')
        - 'Expected_mM': Expected concentration in mM (if parseable from filename)
        
    Examples
    --------
    >>> wav_files = ['0_05MM NADH SPECTRUM.WAV', '1MM PYR SPECTRUM.WAV']
    >>> spectra_df = parse_wav_files(wav_files)
    >>> print(spectra_df.head())
    """
    all_spectra_df = []
    
    for file_name in wav_file_paths:
        # Open the file and read lines to extract wavelength info
        with open(file_name, 'r') as f:
            lines = f.readlines()
        
        # Extract start and end wavelengths from the 8th line (index 7)
        wavelength_info = lines[7].strip().split(',')
        start_wavelength = float(wavelength_info[0])
        end_wavelength = float(wavelength_info[1])
        
        # Load the data from row 9 onwards (skipping the first 8 rows)
        # Select only the first column for Absorbance
        raw_data = pd.read_csv(file_name, skiprows=8, header=None)
        temp_df = pd.DataFrame(raw_data.iloc[:, 0])  # Extract only the first column as Absorbance
        temp_df.columns = ['Absorbance']
        
        # Generate wavelengths based on start, end, and number of data points
        num_points = temp_df.shape[0]
        wavelengths = np.linspace(start_wavelength, end_wavelength, num_points)
        temp_df['Wavelength'] = wavelengths
        
        # Add filename for reference
        temp_df['Filename'] = file_name
        
        # Append to list
        all_spectra_df.append(temp_df)
    
    # Combine all DataFrames
    merged_df = pd.concat(all_spectra_df, ignore_index=True)
    
    # Extract compound and concentration information from filenames
    def parse_filename(filename):
        """Extract compound name and expected concentration from filename."""
        # Example filenames: "0_05MM NADH SPECTRUM.WAV", "1MM PYR SPECTRUM.WAV"
        
        # Extract compound name (NADH, PYR, etc.)
        if 'NADH' in filename.upper():
            compound = 'NADH'
        elif 'PYR' in filename.upper():
            compound = 'PYR'
        elif 'TRIS' in filename.upper():
            compound = 'TRIS'
        else:
            compound = 'UNKNOWN'
        
        # Extract concentration
        # Look for patterns like "0_05MM", "1MM", "10MM", "100MM"
        conc_match = re.search(r'(\d+(?:_\d+)?)\s*MM', filename.upper())
        if conc_match:
            conc_str = conc_match.group(1).replace('_', '.')
            expected_mM = float(conc_str)
        else:
            expected_mM = np.nan
        
        return compound, expected_mM
    
    # Apply parsing to all rows
    merged_df[['Compound', 'Expected_mM']] = merged_df['Filename'].apply(
        lambda x: pd.Series(parse_filename(x))
    )
    
    return merged_df


def extract_spectrum_at_time(
    df: pd.DataFrame,
    target_time: float,
    min_wavelength: float,
    max_wavelength: float,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract spectrum at a specific timepoint from kinetic data.
    
    Finds the row in the DataFrame closest to the target time and extracts
    the spectral data within the specified wavelength range.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time course data with 'Time_s' column and
        wavelength columns (column names should be wavelength values)
        
    target_time : float
        Target time point in seconds
        
    min_wavelength : float
        Minimum wavelength to include (nm)
        
    max_wavelength : float
        Maximum wavelength to include (nm)
        
    verbose : bool, optional (default=True)
        If True, prints the actual time found
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'Wavelength': Wavelength values
        - 'Absorbance': Absorbance values at the target time
        
    Raises
    ------
    ValueError
        If DataFrame doesn't contain 'Time_s' column
        
    Examples
    --------
    >>> kinetic_df = pd.read_csv('timecourse_data.csv')
    >>> spectrum = extract_spectrum_at_time(kinetic_df, target_time=180.0,
    ...                                      min_wavelength=320, max_wavelength=420)
    """
    # Find closest row
    if 'Time_s' not in df.columns:
        raise ValueError("DataFrame must contain 'Time_s' column")
    
    closest_idx = (df['Time_s'] - target_time).abs().idxmin()
    closest_row = df.loc[closest_idx]
    actual_time = closest_row['Time_s']
    
    if verbose:
        print(f"Target Time: {target_time} s")
        print(f"Actual Time Found: {actual_time} s")
    
    # Identify spectral columns (wavelengths)
    non_spectral_cols = ['sample', 'Time_s', 'filename', 'NADH_Conc_SingleWav',
                         'NADH_Conc_Spectral', 'NADH_Method1', 'NADH_Method2',
                         'NADH_Method3', 'NADH_Method4']
    spectral_cols = [c for c in df.columns if c not in non_spectral_cols]
    
    # Build spectrum DataFrame
    spectrum_data = []
    for col in spectral_cols:
        try:
            wavelength = float(col)
            if min_wavelength <= wavelength <= max_wavelength:
                absorbance = closest_row[col]
                spectrum_data.append({
                    'Wavelength': wavelength,
                    'Absorbance': absorbance
                })
        except ValueError:
            # Skip columns that can't be converted to float
            continue
    
    spectrum_df = pd.DataFrame(spectrum_data)
    
    if verbose:
        print(f"Extracted {len(spectrum_df)} wavelengths from {min_wavelength} to {max_wavelength} nm")
    
    return spectrum_df


def load_kinetic_data(
    file_path: str,
    sample_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Load kinetic data from CSV file exported from spectrophotometer.
    
    Parameters
    ----------
    file_path : str
        Path to CSV file containing kinetic data
        
    sample_filter : str, optional
        If provided, filter data for specific sample name
        (e.g., 'CELL_1', 'CELL_2')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with time course spectral data
        
    Examples
    --------
    >>> kinetic_df = load_kinetic_data('data.csv')
    >>> cell_1_df = load_kinetic_data('data.csv', sample_filter='CELL_1')
    """
    df = pd.read_csv(file_path)

    if sample_filter is not None:
        if 'sample' in df.columns:
            df = df[df['sample'] == sample_filter].copy()
            # Reset index after filtering to avoid index issues
            df = df.reset_index(drop=True)
        else:
            print(f"Warning: 'sample' column not found, cannot filter by '{sample_filter}'")

    return df
