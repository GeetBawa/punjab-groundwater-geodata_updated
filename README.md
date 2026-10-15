# punjab-groundwater-geodata_updated


# Punjab groundwater: sparse monitoring data and calibrated uncertainty

Code and data for the manuscript on depth-to-water-level forecasting in Punjab, India,
using well-wise tables from Central Ground Water Board (CGWB) year books (2019-20, 2022-23, 2023-24).

## Files
- punjab_cgwb_dtw_wellwise_parsed.csv : parsed well-wise depth-to-water-level records (m below ground level)
- full_pipeline.py : regenerates every table in the paper (fixed random seed)
- make_figures.py : redraws Figures 1-6 from the pipeline output

## How to run (about 5 minutes)
pip install numpy pandas scipy scikit-learn matplotlib
python full_pipeline.py punjab_cgwb_dtw_wellwise_parsed.csv results
python make_figures.py punjab_cgwb_dtw_wellwise_parsed.csv results figures

## Notes
- The district map in Figure 2 uses approximate district headquarters coordinates.
- The 2022-23 year book contains no records for Patiala district.
- Source data: CGWB year books (Ministry of Jal Shakti, Government of India); the CSV is an extraction from them.
