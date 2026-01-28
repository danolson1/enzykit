# EnzyKit

A collection of tools for processing data related to biochemical enzyme assays, with focus on photodiode array spectrophotometry.

## Installation

### From GitHub (for Google Colab or local use)

```bash
pip install git+https://github.com/danolson1/enzykit.git
```

### For development

```bash
git clone https://github.com/danolson1/enzykit.git
cd enzykit
pip install -e .
```

## Usage in Google Colab

```python
# Install the package
!pip install git+https://github.com/danolson1/enzykit.git

# Import functions
from enzykit import (
    calculate_concentrations,
    process_pdc_timecourse,
    calculate_max_slope,
    load_kinetic_data,
    extract_spectrum_at_time
)

# Load your data
kinetic_df = load_kinetic_data('your_data.csv')

# Process timecourse
results = process_pdc_timecourse(
    spectral_df=kinetic_df,
    standards_df=standards_df,
    assay_start_time=180.0,
    initial_pyruvate_mM=20.0,
    plot=True
)

# Calculate maximum slope
slope_data = calculate_max_slope(
    timecourse_data=results,
    plot=True
)
```

## Modules

### Core Modules

- **spectral.py** - Spectral deconvolution functions for calculating NADH and pyruvate concentrations
- **timecourse.py** - Time course processing for enzyme assays (PDC, FDH, ADH)
- **data_io.py** - Functions for loading and parsing spectrophotometer data files

## Requirements

- pandas >= 1.3.0
- numpy >= 1.20.0
- plotly >= 5.0.0
- statsmodels >= 0.13.0
- scipy >= 1.7.0

## License

MIT License
