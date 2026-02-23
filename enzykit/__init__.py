"""
EnzyKit - Enzyme Assay Analysis Tools

This package provides functions for analyzing spectrophotometric data from photodiode
array spectrophotometers, with focus on enzyme assays including PDC, FDH, and ADH.

Main Functions
--------------
From spectral:
    calculate_concentrations - Calculate NADH and pyruvate concentrations from spectra

From timecourse:
    process_pdc_timecourse - Process PDC enzyme assay time course data
    calculate_max_slope - Calculate maximum slope (velocity) from timecourse data

From data_io:
    parse_wav_files - Parse .WAV standard spectra files
    extract_spectrum_at_time - Extract spectrum at specific timepoint
    load_kinetic_data - Load kinetic CSV data

Example
-------
>>> from enzykit import process_pdc_timecourse, load_kinetic_data
>>>
>>> kinetic_df = load_kinetic_data('data.csv')
>>> results = process_pdc_timecourse(
...     spectral_df=kinetic_df,
...     standards_df=standards_df,
...     assay_start_time=180.0,
...     plot=True
... )
"""

from .spectral import calculate_concentrations
from .timecourse import (
    process_pdc_timecourse,
    select_modeling_data,
    fit_nadh_degradation,
    calculate_max_slope,
)
from .data_io import parse_wav_files, extract_spectrum_at_time, load_kinetic_data

__all__ = [
    'calculate_concentrations',
    'process_pdc_timecourse',
    'select_modeling_data',
    'fit_nadh_degradation',
    'calculate_max_slope',
    'parse_wav_files',
    'extract_spectrum_at_time',
    'load_kinetic_data',
]
