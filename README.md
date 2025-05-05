# Phase Coherence Analysis of Foreshock-Mainshock Sequences

This repository contains Jupyter notebooks for performing waveform analysis and phase coherence calculations on seismic data, particularly focusing on foreshock-mainshock sequences from the San Jacinto Fault Zone (SJFZ), California.
---

## Project Summary

Foreshocks and mainshocks from the same earthquake sequence are typically co-located, meaning their signals share near-identical propagation paths to seismic stations. The phase coherence methods removes path effects through cross-correlation. The inter-station coherence across a range of frequencies is computed and the resulting graph allows an estimation of the spatial extent of nucleation zones to be calculated. The coherence graph should have high coherence at low frequencies and fall off to low coherence at a higher frequency
---

## Contents

### `waveforms.ipynb`
A notebook to compile waveforms for the coherence calculation:
- Retrieves waveforms from data centre
- Use for manual selection of mainshock P-wave arrival

### `coherence_calc.ipynb`
A notebook for coherence calculation
- Implements phase coherence method across selected stations
- Produces coherence graph used to calculate spatial extent
- Includes map and separate seismogram for foreshocks and mainshocks for further analysis

### `synthetic_0extent.ipynb`
A notebook for creating a synthetic earthquake simulation


---

## Example earthquake files

### `filttraceseq11.mseed`
Filtered waveform traces for example earthquake

### `inventory11.xml`
Inventory of stations for example earthquake

### `arrivalseq11.txt`
text file of mainshock P-wave arrival for example earthquake

## Dependencies

### `edphscoh.py`
Custom Python module for computing phase coherence calculation

### `syntheq.py`
Custom Python module for creating synthetic earthquake

### `testappstf.py`
Custon Python module for synthetic apparent source time function

### `San_Jacinto_Fault_2.geojson`
Used for plotting fault map

---
