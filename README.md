# VCM Indicator Species Analysis

This project analyzes species distribution data to identify indicator species and multi-species assemblages that are characteristic of regions experiencing vegetation climate mismatch (VCM). The analysis involves spatial data processing, statistical analysis of species co-occurrence patterns, community detection, and visualization of ecological networks.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Analysis Pipeline](#analysis-pipeline)
- [Running the Analysis](#running-the-analysis)
- [Output Description](#output-description)
- [Additional Notes](#additional-notes)

## Overview

The project implements a complete ecological analysis pipeline for identifying species that indicate VCM habitats using observation data from iNaturalist. It applies multiple analytical methods:

1. **Frequency Analysis**: Identifies species that appear more frequently in VCM habitats
2. **Co-occurrence Analysis**: Measures statistical association between species presence and VCM sites
3. **IndVal (Indicator Value) Analysis**: Calculates specificity and fidelity of species to VCM habitats
4. **Community Analysis**: Detects networks of co-occurring species that collectively indicate VCM
5. **Classifier Evaluation**: Assesses indicator species effectiveness using machine learning

## Project Structure

```
.
├── analysis/
│   ├── cooccurrence_analysis.py    # Co-occurrence statistical analysis
│   ├── community_analysis.py       # Community detection in species networks
│   ├── classifier_evaluation.py    # ML evaluation of indicator species
│   └── frequency_analysis.py       # Species frequency analysis
├── network_analysis/
│   ├── main.py                     # Network analysis main script
│   ├── preprocessing.py            # Network data preparation 
│   ├── graph_construction.py       # Build species interaction networks
│   ├── network_analysis.py         # Network metrics and visualization
│   └── vcm_indicators.py           # VCM indicator extraction from networks
├── processing/
│   ├── processing.py               # Main data processing (lat/long grid)
│   └── gpkg_processing.py          # Alternative processing (raster grid)
├── data_sources/
│   ├── inat-table-for-parthav-alt-lat.csv  # Raw iNaturalist data
│   └── Gymno800mGAM_ecoSN401520.tif        # Raster file for grid assignment
└── outputs/                        # Analysis outputs (created by scripts)
```

## Requirements

This project requires Python 3.7+ and the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn networkx scipy scikit-learn statsmodels geopandas rasterio shapely xgboost python-louvain
```

For visualizing community overlaps with Venn diagrams:
```bash
pip install matplotlib-venn
```

## Analysis Pipeline

The complete analysis follows this sequence:

1. **Data Processing**: Convert raw observation data into a gridded data matrix
2. **Individual Indicator Analysis**:
   - Frequency analysis for species preference for VCM
   - Co-occurrence analysis for species association with VCM
   - IndVal analysis for specificity and fidelity
3. **Network and Community Analysis**:
   - Construct species co-occurrence networks
   - Detect communities of indicator species
4. **Evaluation**:
   - Evaluate effectiveness of indicator species with machine learning
   - Compare individual vs. community indicators

## Running the Analysis

### 1. Data Processing

Process the raw iNaturalist data to create a gridded data matrix:

```bash
python processing/processing.py --input data_sources/inat-table-for-parthav-alt-lat.csv --output inat-data-matrix-latlong.csv --altitude 2500 --top-species 300 --min-species-count 100
```

### 2. Frequency Analysis

Analyze species occurrence frequency in VCM vs. non-VCM sites:

```bash
mkdir -p outputs/frequency_analysis
python analysis/frequency_analysis.py --input inat-data-matrix-latlong.csv --output-dir outputs/frequency_analysis
```

### 3. Co-occurrence Analysis

Calculate statistical associations between species and VCM habitats:

```bash
mkdir -p outputs/cooccurrence_analysis
python analysis/cooccurrence_analysis.py --input inat-data-matrix-latlong.csv --output-prefix outputs/cooccurrence_analysis/
```

### 4. IndVal Analysis

Calculate indicator values (specificity and fidelity):

```bash
mkdir -p outputs/indval_analysis
python analysis/indval_analysis.py --input inat-data-matrix-latlong.csv --output-dir outputs/indval_analysis
```

### 5. Community Analysis

Detect communities of co-occurring species as multi-species indicators:

```bash
mkdir -p outputs/community_analysis
python analysis/community_analysis.py --data-matrix inat-data-matrix-latlong.csv --indval outputs/indval_analysis/indval_full_results.csv --cooccurrence outputs/cooccurrence_analysis/cooccurrence_full_results.csv --frequency outputs/frequency_analysis/species_frequency_analysis.csv --output-dir outputs/community_analysis --resolution 0.4
```

### 6. Classifier Evaluation

Evaluate the effectiveness of indicator species using machine learning:

```bash
mkdir -p outputs/classifier_analysis
python analysis/classifier_evaluation.py --input inat-data-matrix-latlong.csv --output-dir outputs/classifier_analysis
```

## Output Description

Each analysis produces specific outputs:

### Frequency Analysis
- `species_frequency_analysis.csv`: Statistics on species occurrence frequencies
- `significant_vcm_preferences.csv`: Species with significant VCM preferences
- Visualizations of frequency distributions and top indicators

### Co-occurrence Analysis
- `cooccurrence_full_results.csv`: Full statistical metrics for all species
- `top_indicators_summary.csv`: Summary of top 15 indicator species
- Various visualizations of co-occurrence patterns and effect sizes

### IndVal Analysis
- `indval_full_results.csv`: Complete IndVal scores for all species
- `top_indval_summary.csv`: Summary of top indicator species by IndVal
- Visualizations of indicator species characteristics

### Community Analysis
- `species_cooccurrence_matrix.csv`: Matrix of species co-occurrences
- `community_metrics.csv`: Metrics for each detected community
- `community_species_report.csv`: Detailed species listing by community
- Network visualizations and community metric comparisons

### Classifier Analysis
- Performance metrics comparing baseline vs. indicator species models
- ROC curves and confusion matrices
- Feature importance visualizations
- Indicator species overlap analysis

## Additional Notes

- The analysis can be computationally intensive for large datasets. Consider using a subset of the data for testing.
- Community detection resolution parameter (default 0.4) can be adjusted to detect more or fewer communities.
- Alternative grid assignment using raster files can be performed with `gpkg_processing.py` instead of `processing.py`.
- All scripts support command-line arguments for customizing file paths and analysis parameters.
- The project assumes the structure of the iNaturalist data includes taxonomic columns, coordinates, and VCM classification.

For further customization, each script contains detailed documentation and can be modified to accommodate different data structures or analytical approaches.
