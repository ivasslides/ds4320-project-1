# DS4320 Project 1: ...TITLE....

Executive Summary
///........//// 

| Spec | Value |
| :--- | :--- |
| Name | Iliana Vasslides | 
| NetID | fbv2sc | 
| DOI | ..... |
| Press Release | [link to Press Release](https://github.com/ivasslides/ds4320-project-1/blob/main/press_release.md) | 
| Data | [link to data folder](https://myuva-my.sharepoint.com/:f:/g/personal/fbv2sc_virginia_edu/IgD0_dw8bOXbQqtcXuFmWMcQAbQHJMFY-ToaHPTsxSS_jvY?e=cAJDgh)  | 
| Pipeline | .... | 
| License | ... | 


## Problem Definition
#### General Problem 
* Forecasting energy demands 
#### Specific Problem
* How accurately can machine learning models forecast electricity demand during extreme temperature events compared to during normal weather conditions in Virginia? 

#### Rationale 
The purpose of this refinement is to look more closely at energy prediction when the weather pattern strays from normal, and therefore the energy demand might stray from normal as well. During extreme temperature events, such as heat waves and cold snaps, citizens tend to use more electricity than usual to keep their homes at the desired temperature. This increased use of heating or air-conditioning can cause sudden spikes in electricity demand that are more difficult for forecasting models to predict accurately. By focusing specifically on these extreme conditions, the analysis can evaluate whether traditional prediction models still perform well when demand patterns become irregular. Understanding these differences can help improve forecasting methods and ensure that energy providers are better prepared for periods of unusually high demand.


#### Motivation
Electricity companies want to be as prepared as possible to best serve their clients. With this in mind, it would be beneficial if they were able to accurately predict when large changes in energy demands will occur. Accurate forecasts allow utilities to plan how much electricity needs to be generated and distributed across the grid at any given time. If demand is underestimated during extreme weather events, it can strain the power grid and potentially lead to outages or emergency measures. On the other hand, overestimating demand can result in unnecessary costs from generating excess electricity. By improving predictions of demand during extreme temperature events, energy providers can make more informed operational decisions and maintain a more reliable and efficient power system.


#### Press Release 
[Let's hope energy can keep up with Mother Nature's big changes](https://github.com/ivasslides/ds4320-project-1/blob/main/press_release.md) 

## Domain Exposition 
#### Terminology 
| Term | Definition |
| :--- | :--- | 
| *Daily Average Dry Bulb Temperature* | the average temperature of the air measured by a therometer that is shielded for radiation and moisture|
| *Electricity demand* | the total amount of electrical power being used at a specific moment, reflecting the instantaneous load on the power grid | 
| *Extreme weather event* | rare weather occurrence that is outside of the range of average weather patterns for a particular location and time of year |
| *Linear regression model* | fundamental supervised machine learning algorithm used to model the relationship between a dependent variable and one or more indepedent variables | 
| *MWH* | megawatt-hour; a unit of energy representing one million watt-hours; used to measure electricity consumption or generation over time| 
| *NOAA* | National Oceanic and Atmospheric Administration; U.S. federal agency that studies and monitors the oceans, atmosphere, and coastal areas to provide critical information to the public | 
| *Random forest model* | ensemble machine learning algorithm that builds multiple decision trees and combines their outputs to improve prediction accuracy| 

#### Project Domain
The main domain that this project lives in is Demand Forecasting Analytics. Demand forecasting analytics focuses on understanding the different factors that influence electricity consumption at certain times. By combining historical electricity consumption data with external variables such as weather conditions, analysts can identify relationships to accurately predict future energy demands. These insights allow energy companies to make informed decisions about generation and distribution. Additionally, anticipating unusual demand patterns is an important aspect of this domain. By analyzing how weather variability affects energy usage, forecasting models can help energy companies better prepare for these scenarios.

#### Background Readings 
[link to OneDrive folder](https://myuva-my.sharepoint.com/:f:/g/personal/fbv2sc_virginia_edu/IgBT-8V3aSw6Ra1ayGtJOW_mAbEVKdL8xiyp1iE4mzLFmzA?e=r1OJln) 

#### Summary of Readings 
| Title | Brief Description | Link to File |
| :--- | :--- | :--- |
|*Allocation of policy resources for energy storage development <br> considering the Inflation Reduction Act* | This paper touches on the issue surrounding gas emissions from energy storage areas, and how the Inflation Reduction Act <br> has impacted things. It examines the options for different regions, and the tax and emissions benefits for each. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQAobMfcg-0YTpssH-BMO19uAcW-IEkpGgbGLEM_FBrCyBg?e=GsZfPE) |
|*Extreme weather events on energy systems* | This paper examines the impacts of extreme weather events on energy systems and their associated infrastructures. It reviews <br> published studies to find a solution to help energy systems maintain regular operations. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQAMSpiPuU2sRIKLnHiKZIKfAasXiMpPlPZS9cxc60p2Qc0?e=btj1xF) |
| *How does extreme weather impact energy demand and energy rates* | This article explains why and which weather events have the biggest impact on energy demands and the prices. It also gives <br>advice to help clients keep costs down. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQDQYKV7yPiCSLWQR4xDmNX9AY8Qgi98dtROQOk6OTBGtNU?e=ACKCzt) |
| *Keeping the lights on in our neighborhoods during power outages* | This blog dives into recent microgrid projects that have been started in a variety of states to help with electricity demands. <br> The goal of these projects, funded by the DOE in 2023, is to increase resilience during major events and reduce power outages. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQD_8rKMuosQTrTh9D7Nf-vIAaFrn93OXRWaAcmbzXBnwKw?e=79zTia) |
| *NOAA Local climatological data datase documentation* | This is the offica documentation from the NOAA local climatological dataset (LCD), where the weather data for this project was requested from. It defines the variables in the dataset and how they are collected. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQBBpSWIgpgNRqHtKWDpZSDrAX6YE6JG0rFcJOeMhdzvUYs?e=vZ74iX)

## Data Creation 
#### Raw Data Acquistition 
...

#### Code Used
*table* 

#### Bias Identificaton 
... 

#### Bias Mitigation
... 

#### Rationale 
.... 


## Metadata
#### Schema
... 

#### Data Table 
*table* 

#### Data Dictionary 
*table* 

#### Uncertainty 
... 

to input code 
```python
# code here
import pandas as pd
``` 
