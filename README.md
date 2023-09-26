## Graph Construction on Complex Spatio-Temporal Data for Enhancing Graph Neural Network-Based Approaches

#### Authors: Stefan Bloemheuvel, Jurgen van den Hoogen and Martin Atzmueller

![The proposed framework](location_or_data.png)

## Streamlit Demo Page

[Link to Streamlit demo page](https://stefanbloemheuvel-graph-comparison-streamlit-app-dplan6.streamlit.app/)

## Data
data can be found at = https://zenodo.org/record/7900964

## Requirements
* tensorflow
* tsl
* numpy
* pandas
* cuda
* networkx
  
## Usage
the inputs_la, input_bay, inputs_ci and inputs_cw files should be put in the sensor locations folder.
in a 'data' folder the whole data.zip file should be placed.
run either the earthquake.py file or traffic.py file for the results.
models for the forecasting traffic analysis can be found in all_models.py

## Results 
The resulting graphs and their characteristics:
| Method         | CI (741)                | CW (741)                | METR-LA (21,321)                  | PEMS-BAY (52,650)                  |
|----------------|-------------------------|-------------------------|-----------------------------------|------------------------------------|
|                | Edges     (Density)     | Edges     (Density)     | Edges              (Density)      | Edges              (Density)       |
| **Signal**     |                         |                         |                                   |                                    |
| Correlation    | 195 (26.3%)             | 42 (5.7%)               | 13,478 (63.2%)                    | 35,644 (67.7%)                     |
| DTW            | 536 (72.3%)             | 723 (97.6%)             | 21,365 (100.2%)                   | 26,287 (49.9%)                     |
| MIC            | 739 (99.7%)             | 697 (94.1%)             | 21,315 (100.0%)                   | 52,950 (100.6%)                    |
| **Location**   |                         |                         |                                   |                                    |
| Gabriel        | 69 (9.3%)               | 66 (8.9%)               | 252 (1.2%)                        | 410 (0.8%)                         |
| Gaussian       | 89 (12.0%)              | 275 (37.1%)             | 12,557 (58.9%)                    | 14,527 (27.6%)                     |
| KNN            | 212 (28.6%)             | 439 (59.2%)             | 3,800 (17.8%)                     | 6,816 (12.9%)                      |
| KNN-W          | 672 (90.7%)             | 640 (86.4%)             | 3,118 (14.6%)                     | 7,030 (13.4%)                      |
| K-means        | 48 (6.5%)               | 42 (5.7%)               | 3,097 (14.5%)                     | 8,025 (15.2%)                      |
| MinMax         | 618 (83.4%)             | 533 (71.9%)             | 18,371 (86.2%)                    | 38,098 (72.4%)                     |
| Optics         | 780 (105.3%)            | 91 (12.3%)              | 2,110 (9.9%)                      | 3,144 (6.0%)                       |
| RNG            | 105 (14.2%)             | 101 (13.6%)             | 591 (2.8%)                        | 938 (1.8%)                         |


The resulting MAE and MSE scores on 4 datasets, where CI and CW are time series regression, and Metr-LA and Pems-Bay are forecasting:

| Method         |               | CI (MAE) | CI (MSE) | CW (MAE) | CW (MSE) | METR-LA (MAE) | METR-LA (MSE) | PEMS-BAY (MAE) | PEMS-BAY (MSE) |
|----------------|---------------|----------|----------|----------|----------|---------------|---------------|----------------|----------------|
| **Signal**     | Correlation   | 0.31     | 0.20     | **0.37** | **0.23** | 3.65          | 54.43         | **1.84**       | 18.49          |
|                | DTW           | 0.32     | 0.21     | 0.39     | 0.25     | 3.65          | 54.20         | 1.85           | **18.46**      |
|                | MIC           | **0.30** | **0.19** | 0.39     | 0.25     | **3.64**      | 53.99         | 1.85           | 18.53          |
|                |               |          |          |          |          |               |               |                |                |
| **Location**   | Gaussian      | 0.34     | 0.24     | 0.42     | 0.28     | 3.66          | 54.16         | 1.86           | 18.80          |
|                | MinMax        | 0.31     | 0.21     | 0.41     | 0.26     | 3.64          | **53.74**     | 1.85           | 18.48          |
|                | KNN           | 0.31     | 0.20     | 0.39     | 0.24     | 3.68          | 54.94         | 1.87           | 18.74          |
|                | KNN-W         | 0.32     | 0.22     | 0.38     | 0.23     | 3.69          | 54.95         | 1.88           | 18.93          |
|                | Kmeans        | 0.34     | 0.23     | 0.43     | 0.30     | 3.69          | 55.65         | 1.88           | 19.44          |
|                | Optics        | 0.32     | 0.21     | 0.39     | 0.25     | 3.73          | 57.24         | 1.93           | 20.65          |
|                | Gabriel       | 0.36     | 0.28     | 0.46     | 0.34     | 3.78          | 59.06         | 1.94           | 20.96          |
|                | RNG           | 0.37     | 0.28     | 0.46     | 0.33     | 3.75          | 58.05         | 1.91           | 20.38          |
| **cv ($\mu / \sigma$)** |      | 6.6%     | 13.4%    | 7.1%     | 14.4%    | 1.3%          | 3.3%          | 1.9%           | 4.9%           |


You can copy and paste this Markdown code into your GitHub page, and it should display the table correctly.
