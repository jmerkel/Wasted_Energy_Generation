# Wasted_Energy_Generation
Use Machine Learning to predict energy generation &amp; energy wasted

## Presentation
### Presentation Link
[Presentation](https://docs.google.com/presentation/d/1yfZyfvpHZm6fd5A6OndYg5I7bu-Obim3QkOrRmsWnZM/edit?usp=sharing)

### Topic
From the perspective of a power generation/utility company, we are identifying the weather conditions that cause an excessive amount of energy that was generated to be wasted.

### Reason why selected topic
Many industrial companies and regional Utilities are using Big Data tools (such as Historical Data Archives) to record and analyze equipment data. This is being done in an effort to optimize production, reduce maintenance time, cut unnecessary costs, and identify safety/pollution risks.

By predicting which combination of weather features leads to excessive energy generation waste, we are solving a common situation many utilities and energy companies face: how to determine which power-plants need to be running or staffed and if the generation (and pollution) from those plants are necessary to meet power demands.


### Description of data source
The data used in this model is based in Spain. The Kaggle compiled dataset uses energy production/demand data from the ENTSOE Public portal, while the weather data is sourced from the Open Weather API and focuses on the 5 largest Spanish Cities. In both cases, each data source takes readings hourly between 2015 through 2019.

The weather dataset contains information for each city such as temperatures, pressure, wind speed, rain, clouds and description. The energy dataset contains energy generation readings from different types of generators (Biomass, Wind, Fossil Fuel, etc.), forecasted energy generation, actual demand, and price.

### Questions hope to answer
- Can we predict if the current configuration of energy production will waste over 300 MW of energy based on weather conditions?
- Can we predict specific times or days that will require out of the ordinary amounts of power?

### Description of Data Exploration


### Description of Analysis Phase



## GitHub
### Description of the communication protocols

### Outline of the project
- Import Data
- Clean
- Identify wasted rows (1 or 0)
- Store in DB
- Machine Learning


## Machine Learning
### Description of Prelim Data Preprocessing

### Description of Preliminary feature selection (and reasoning)

### Description of how data was split into training and testing sets

### Explanation of model choice w/ Limitations/Benefits




## Database



## Dashboard (Website, Screenshot, etc.)
### Description of tool that will be used to create final dashboard
### Description of interactive elements



## References
[Peaking power plant](https://en.wikipedia.org/wiki/Peaking_power_plant)
[Load following power plant](https://en.wikipedia.org/wiki/Load_following_power_plant)
[Startup Time](https://www.wartsila.com/energy/learn-more/technical-comparisons/combustion-engine-vs-gas-turbine-startup-time)
[Costs](https://www.eia.gov/tools/faqs/faq.php?id=487&t=3)
