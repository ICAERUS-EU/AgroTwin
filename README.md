<div align="center">
  <p>
    <a href="https://icaerus.eu" target="_blank">
      <img width="50%" src="https://icaerus.eu/wp-content/uploads/2022/09/ICAERUS-logo-white.svg"></a>
    <h3 align="center">AGROTWIN</h3>
    
   <p align="center">
    A drone based agronomic DSS to create and analyse vineyards digital twins and optimize pesticide treatments.
    <br/>
    <br/>
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/icaerus-repo-template/issues">Report Bug</a>
    -
    <a href="https://github.com/icaerus-eu/icaerus-repo-template/issues">Request Feature</a>
  </p>
</p>
</div>

![Downloads](https://img.shields.io/github/downloads/icaerus-eu/icaerus-repo-template/total) ![Contributors](https://img.shields.io/github/contributors/icaerus-eu/icaerus-repo-template?color=dark-green) ![Forks](https://img.shields.io/github/forks/icaerus-eu/icaerus-repo-template?style=social) ![Stargazers](https://img.shields.io/github/stars/icaerus-eu/icaerus-repo-template?style=social) ![Issues](https://img.shields.io/github/issues/icaerus-eu/icaerus-repo-template) 

## Table Of Contents

* [Summary](#summary)
* [Features](#features)
  * [Select inputs](#select-inputs)
  * [Vine extraction](#vine-extraction)
  * [Soil modeling](#soil-modeling)
  * [Canopy density](#canopy-density)
  * [Canopy thickness, height and volume](#canopy-thickness-height-and-volume)
  * [Optimum volume rate (L/ha)](#optimum-volume-rate-lha)
  * [Results](#results)
  * [Output Maps (LAI, Volume Rate)](#output-maps-lai-volume-rate)
* [Installation](#installation)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)
  
## Summary
This project leverages UAV image analysis to revolutionize viticulture by integrating cutting-edge technology, AI, and precision agriculture. Specifically, it utilizes 3D point clouds generated from consumer-grade RGB drones to develop a Decision Support System (DSS) driven by python algorithms. These algorithms will automatically analyze digital twins of vineyards given specific inputs to assess canopy biometrics and field parameters, creating vigor and prescription maps for optimized variable rate pesticide treatments.

## Features

### Select inputs

For the correct elaboration we need to:
- Read the elaborated point cloud (.ply)
- Read the excel file with coordinates of the interesting points (2 points for each vine, starting point and ending point)
- Select the phenological phase
- Select the sprayer type
- Insert the inter-row distance (m)
  
![pointcloud image](https://github.com/Agrobitsrl/AgroTwin/blob/main/docs/images/pointcloud.png?height=50)

### Vine extraction

- Define the line passing through each selected vine
- Translate the point cloud to vine portion
- Rotate the point cloud to align the canopy to x-axis
- Extract the vine in the interested bounds
  
![extracted vine](https://github.com/Agrobitsrl/AgroTwin/blob/main/docs/images/Extracted_portion.png?height=10)

### Soil modeling
In the case of terracing, two adjacent inter-row paths can have different elevations.
We need to find the relative heights of the vine points with respect to a plane that approximates the soil surface.

Analysis:
- Select points from the point cloud that fall within specific ranges in the y and z dimensions (positive and negative). 
- The selection criteria are combined into a boolean mask, and then the points that meet these criteria are extracted (S1 and S2).
- Calculate the centroid of S1 and S2
- Compare the z-coordinates of the centroids to determine which subset of soil points (S1 or S2) is lower.
- The objective function to be minimized helps in fitting a plane to the soil points.
- The minimize function is used to find the best-fit plane by minimizing the objective function. 
- Calculate the relative heights of vine portion point cloud with respect to soil plane

### Canopy density
Calculate the canopy density by dividing the canopy region into a grid, counting the points in each cell, normalizing the density, and then computing a density descriptor based on a threshold.

Analysis:
- Setting the grid dimensions
- Define the region of interest (ROI) for the vine canopy in the x, y, and z dimensions
- Calculate Density in each grid cell

### Canopy thickness, height and volume
Analysis:
- Calculate the 95th percentile of the height (z-coordinate) to estimate the canopy height.
- Compute the canopy thickness by subtracting the 10th percentile from the 90th percentile (y-coordinate)
- Calculate the canopy height relative to the minimum z-coordinate of the vine.
- Compute the convex hull of the canopy points, which is the smallest convex shape that encompasses all the points.

### Optimum volume rate (L/ha)
Analysis:
- Find the appropriate height dosing value (h_dosa3d) and the efficiency factor (EC_h) based on the canopy height
- Find the appropriate width dosing (w_dosa3d) and the efficiency factor (EC_w) value based on the canopy thickness
- Determine p Value Based on BBCH Stage and Sprayer Performance Factors
- Calculate outputs (LAI, TRV, LWA, Volume rate)

### Results
- After computing, the algorithm returns a csv (Results.csv) with all the parameters calculated:

![results](https://github.com/Agrobitsrl/AgroTwin/blob/main/docs/images/results.png?height=50)

### Output Maps (LAI, Volume Rate)
- Create maps of LAI and optimized V_rate in .shp and .png format from the Results.csv file

![Output](https://github.com/Agrobitsrl/AgroTwin/blob/main/docs/images/input_V_r.png?height=50)

## Installation
It is built upon the work of `scipy`, `open3d` and a few more. For all the libraries see requirements.txt.

## Authors
* **Simone Kartsiotis** - *Agrobit srl* 
* **Antonio Donnangelo** - *Agrobit srl* - [Antonio Donnangelo](https://github.com/AntonioDonnangelo)

## Acknowledgements
This project is funded by the European Union, grant ID 101060643.

<img src="https://rea.ec.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2021-04/EN-Funded%20by%20the%20EU-POS.jpg" alt="https://cordis.europa.eu/project/id/101060643" width="200"/>
