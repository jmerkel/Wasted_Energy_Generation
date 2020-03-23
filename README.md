# Wasted_Energy_Generation
Use Machine Learning to predict energy generation &amp; energy wasted

## Presentation
[Presentation](https://docs.google.com/presentation/d/1yfZyfvpHZm6fd5A6OndYg5I7bu-Obim3QkOrRmsWnZM/edit?usp=sharing)

## Dashboard
[Presentation](https://docs.google.com/presentation/d/1yfZyfvpHZm6fd5A6OndYg5I7bu-Obim3QkOrRmsWnZM/edit?usp=sharing)


### Topic
From the perspective of a power generation/utility company, we are attempting to determine if it is possible to identify & predict situations where an excessive amount of energy is generated due to weather various weather conditions.


### Reason why selected topic
Many industrial companies and regional Utilities are using Big Data tools (such as Historical Data Archives) to record and analyze equipment data. This is being done in an effort to optimize production, reduce maintenance time, cut unnecessary costs, and identify safety/pollution risks.

By predicting which combination of weather features leads to wasted energy generation, we are solving a common problem many utilities and energy companies face: how to determine which power-plants need to be operational and if the generation (and pollution) from those plants are more than necessary to meet power demands.


### Description of data source
The data used in this model is based in Spain. The Kaggle compiled dataset uses energy production/demand data from the ENTSOE Public portal. The weather data is sourced from the Open Weather API and focuses on the 5 largest Spanish Cities. In both cases, each data source takes readings hourly between 2015 through 2019.

The weather dataset contains information for each city such as temperatures, pressure, wind speed, rain, clouds and description. The energy dataset contains energy generation readings from different types of generators (Biomass, Wind, Fossil Fuel, etc.), forecasted energy generation, actual demand, and price.


### Questions hope to answer
- Can we predict if the current configuration of energy production will waste over 300 MW of energy based on weather conditions?
- Can we predict specific times or days that will require out of the ordinary amounts of power?


### Description of Data Exploration
Initial data exploration was to use Excel and Python. Excel was used to review a few columns where data appeared to be empty or 0. Python was used to pull unique values, data ranges, and view DateTime values. By doing this, we were able to ensure our data could be cleaned efficiently & completely. Furthermore, this helped identify a few potential errors, such as a key column (city name) having a value with leading spaces which could cause data saving & retrieval problems.


### Description of Analysis Phase
For additional information, refer to the Machine Learning section of this ReadMe.

After data cleaning, we tested a few different classification models to determine to most appropriate. Judging from the accuracy score and other metrics, the SVM RBF narrowly beat out the other models types.

Using this model, we used one city as the "training" set, and the other cities as the testing set.


### Technologies, languages, tools
The following tools were used:
- Excel
- Python: Pandas, sklearn, sqlalchemy, plotyly, hvplot
- Machine Learning: Logistic Regression, SVM, Metrics


### Result of Analysis
Our model had some minor success in identifying situations where an excessive amount of energy is wasted. At very high percentages, it identified non-wasted situations correctly. However, it only picked up the wasted energy generation situations at rates of 10 - 20%.

This is of some benefit, as the model is "safe". If it didn't have a high recall number for non-wasted energy situations, this model would not be used, as energy MUST be supplied adequately. Even with the low scores of the wasted energy identification, this model may be able to reduce high/cost equipment rates by 10%

The model also was less accurate when used for Bilbao.


### Recommendation for future analysis
- Use a different threshold for "wasted energy" (> 310 MW, ~200 MW, etc.)
- Identify exact weather conditions that "lead to" wasted energy generation
- Identify why the Bilbao prediction was not as accurate
- Instead of classifying data, attempt to predict energy generation as well the type of energy


### Done Differently
- Consider utilizing a user interface
- Splitting the cleaning and Machine Learning components into completely separate scripts
- Utilize SQLite more
- Not be disrupted by Corona Virus


## GitHub
### Outline of the project
- Import Data
- Clean data sets
- Identify wasted rows (1 or 0)
- Merge data sets into City - Weather/Energy tables
- Store data in SQlite
- Identify Machine Learning Model to use
- Use Machine Learning Model to make predictions for each city
- Generate Metrics for each city
- Generate graphs and pictures


## Machine Learning
### Description of Data Preprocessing
Because this is time series data, we had to take a few steps:
- convert all DateTimes to the DateTime format, convert to UTC, and convert to Ordinal times (integer)
- Remove Descriptions of weather to only use numerical data
- Scale the data. The data varies in each feature from very high numbers to very small or 0 numbers.


### Description of feature selection (and reasoning)
For the model, we selected all numerical weather features as well as the date-time, forecast for the day ahead, forecast of the wind onshore day ahead, total load forecast, and price day ahead features.

These items were picked because they are numerical and forecasts. By selecting these features, we can have users create data tables with forecasted weather/data and be able to input this information into the model.


### Description of how data was split into training and testing sets
To test model types, data from one city (Madrid) was split into train & test sets. After the model was

### Explanation of model choice w/ Limitations/Benefits
SVM - RBF
Benefits: The model is simple & easy to understand. The model can be quickly used and implemented to help identify wasted energy situations.

Limitations:  

### Description and explanation of models confusion matrix including accuracy score
Model - SVM RBF: Creating a hyperplane between the types. RBF is the type of hyperplane kernal used. Unlike Linear or Poly, RBF can be multiple individual areas
Confusion Matrix: Accurate models would want values to be high in the True/True & False/False sections.
- 0:[True/True, True/False]
- 1:[False/True, False/False]
Accuracy Score: Determines how often the classifier predicts correctly
Classification Report:
- Precision: The accuracy of predicting the correct result
- Recall:  The ability to detect the type in question
- F1 score: Single summary Statistic

Our model had Accuracy scores in ~75%, with a high F1 score and very high Recall scores for non-wasted energy situations, but low F1 score and very low recall scores for wasted energy situations.

### Addressed the question or problem the team is solving?
Our model was somewhat successful in identifying wasted energy situations. Though not highly accurate, it was "safe" in not identifying lower margin situations.


### Statistical Analysis for next phase of project?
- Are the situations where energy generation is wasted continuous or occur sporadically
- Predict the energy generated and energy wasted
- Compare predicted energy generation and predicted energy wasted, with the actual values. Identify if the predictions are accurate.


## References
- [Peaking power plant](https://en.wikipedia.org/wiki/Peaking_power_plant)
- [Load following power plant](https://en.wikipedia.org/wiki/Load_following_power_plant)
- [Startup Time](https://www.wartsila.com/energy/learn-more/technical-comparisons/combustion-engine-vs-gas-turbine-startup-time)
- [Costs](https://www.eia.gov/tools/faqs/faq.php?id=487&t=3)
