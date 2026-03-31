# Predicting Energy Usage Pipeline

*Starting the pipeline from the analysis section, since the data tables that are stored in duckdb are already cleaned and ready for processing.* 


## Analysis 
```python 
####### imports and logging 
import pandas as pd 
import duckdb 
import logging 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='analysis.log'
)
logger = logging.getLogger(__name__)

####### load duckdb tables into pandas dfs 
con = None
try: 
    # create and verify connection 
    con = duckdb.connect(database='project1.db', read_only=False) 
    logger.info("Connected to duckdb instance.")

    # inserting tables 
    cville = con.execute(f"""
        SELECT * FROM cville;
    """).fetchdf()
    dulles = con.execute(f"""
        SELECT * FROM dulles;
    """).fetchdf()
    lynchburg = con.execute(f"""
        SELECT * FROM lynchburg;
    """).fetchdf()
    norfolk = con.execute(f"""
        SELECT * FROM norfolk;
    """).fetchdf()
    dom = con.execute(f"""
        SELECT * FROM dom;
    """).fetchdf()

    logger.info("All tables loaded into pandas dataframes")

except Exception as e:
    logger.error(f"An error occurred: {e}")
``` 

Before starting the analysis, the 4 weather datasets and energy datasets were merged for easier analysis. They were merged on the date columns, and then the `Datetime` column from the DOM_hourly.csv was dropped as to not have repetitive columns.

```python
####### combine weather data into one df
weatherdf = pd.concat([cville, dulles, lynchburg, norfolk], ignore_index=True)

# ensure both Date columns are in proper format before merging 
weatherdf['DATE'] = pd.to_datetime(weatherdf['DATE'])
dom['Datetime'] = pd.to_datetime(dom['Datetime'])
# round to the hour to prevent errors
dom['Datetime'] = dom['Datetime'].dt.floor('h')

# merge weather and energy dfs 
combined_df = pd.merge(weatherdf, dom, left_on='DATE', right_on='Datetime', how='inner')
combined_df = combined_df.drop(columns=['Datetime'])
logger.info("Weather and energy dataframes merged into combined_df.")

####### save combined_df as parquet for later use
combined_df.to_parquet("data/combined_df.parquet", engine='pyarrow', index=False)
logger.info("combined_df saved as parquet")

# save combined_df to duckdb table for later press release use 
con = None
try: 
    # create and verify connection 
    con = duckdb.connect(database='project1.db', read_only=False) 
    logger.info("Connected to duckdb instance.") 

    # inserting table 
    con.execute(f"""
        DROP TABLE IF EXISTS combined_df;
        CREATE TABLE combined_df AS
        SELECT * FROM read_parquet('data/combined_df.parquet');
    """)

    logger.info("combined_df loaded into duckdb table.")

except Exception as e:
    logger.error(f"An error occurred: {e}")
``` 

For the `Station` column, one-hot encoding was used so that the different cities could be added as a feature for analysis. When forming the X dataframe, unnecessary columns were dropped such as: `Date`, because using other more specific date columns; `DOM_MW`, because is a target variable; `Sunrise/Sunset`, because are not as relevant to the analysis; and `DailyWeather`, because the `EWE` column already accounts for this. 

A train-test split with 80/20 split was used, with the target variable being `DOM_MW`, aka the energy usage. 


```python 
####### one-hot encoding 
df = pd.get_dummies(combined_df, columns=['STATION'], drop_first=False)

# dropping columns
X = df.drop(columns=['DATE', 'DOM_MW', 'Sunrise', 'Sunset', 'DailyWeather'])
# target variable is DOM_MW
y = df['DOM_MW']

# Split data into train and test sets
X_train, X_test, y_train, y_test, extreme_train, extreme_test = train_test_split(
    X, y, df['EWE'], test_size=0.2, random_state=42)
``` 

A linear regression model and random forest model were used to predict energy usage based on the weather data. The goal with these two models was to have a simpler and them more complex prediction models to see if there were significant differences in the performance metrics. For each model, root mean square error (RMSE), mean absolute error (MAE), and r-sqaured (R^2) were calculated for the performance overall, performance on normal weather conditions, and performance on the extreme weather conditions. Using these three metrics shows the error overall (MAE), the error with large errors being penalized more (RMSE), and the variance in the model (R^2) The results were combined into a pandas dataframe and stored in a duckdb table for later use during the visualization stage.

**Linear Regression model**
```python
####### linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Overall performance for lr model 
lr_overall_rmse = round(mean_squared_error(y_test, y_pred_lr) ** 0.5, 4)
lr_overall_mae = round(mean_absolute_error(y_test, y_pred_lr), 4)
lr_overall_r2 = round(r2_score(y_test, y_pred_lr), 4)

# Performance by weather type
normal_mask = extreme_test == 0
extreme_mask = extreme_test == 1

# Normal weather
lr_n_rmse = round(mean_squared_error(y_test[normal_mask], y_pred_lr[normal_mask]) ** 0.5, 4)
lr_n_mae = round(mean_absolute_error(y_test[normal_mask], y_pred_lr[normal_mask]), 4)
lr_n_r2 = round(r2_score(y_test[normal_mask], y_pred_lr[normal_mask]), 4)

# Extreme weather
lr_e_rmse = round(mean_squared_error(y_test[extreme_mask], y_pred_lr[extreme_mask]) ** 0.5, 4)
lr_e_mae = round(mean_absolute_error(y_test[extreme_mask], y_pred_lr[extreme_mask]), 4)
lr_e_r2 = round(r2_score(y_test[extreme_mask], y_pred_lr[extreme_mask]), 4)

# put into a dictionary for easy addition to df later 
lr_results = {
    'Model': 'Linear_Regression',
    'Overall_RMSE': lr_overall_rmse,
    'Normal_RMSE': lr_n_rmse,
    'Extreme_RMSE': lr_e_rmse,
    'Overall_MAE': lr_overall_mae,
    'Normal_MAE': lr_n_mae,
    'Extreme_MAE': lr_e_mae,
    'Overall_R2': lr_overall_r2,
    'Normal_R2': lr_n_r2, 
    'Extreme_R2': lr_e_r2}

logger.info('Linear regression model done.')
```

**Random Forest model** 
```python
######## training random forest model, with 100 trees 
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# predict on the test set
y_pred = rf.predict(X_test)
# Overall performance
rf_overall_rmse = round(mean_squared_error(y_test, y_pred) ** 0.5, 4)
rf_overall_mae = round(mean_absolute_error(y_test, y_pred), 4)
rf_overall_r2 = round(r2_score(y_test, y_pred), 4)

# Performance by weather type
normal_mask = extreme_test == 0
extreme_mask = extreme_test == 1

# Normal weather
rf_n_rmse = round(mean_squared_error(y_test[normal_mask], y_pred[normal_mask]) ** 0.5, 4)
rf_n_mae = round(mean_absolute_error(y_test[normal_mask], y_pred[normal_mask]), 4)
rf_n_r2 = round(r2_score(y_test[normal_mask], y_pred[normal_mask]), 4)
# Extreme weather
rf_e_rmse = round(mean_squared_error(y_test[extreme_mask], y_pred[extreme_mask]) ** 0.5, 4)
rf_e_mae = round(mean_absolute_error(y_test[extreme_mask], y_pred[extreme_mask]), 4)
rf_e_r2 = round(r2_score(y_test[extreme_mask], y_pred[extreme_mask]), 4)

# put into a dictionary for easy addition to df later
rf_results = {
    'Model': 'Random_Forest',
    'Overall_RMSE': rf_overall_rmse,
    'Normal_RMSE': rf_n_rmse,
    'Extreme_RMSE': rf_e_rmse,
    'Overall_MAE': rf_overall_mae,
    'Normal_MAE': rf_n_mae,
    'Extreme_MAE': rf_e_mae,
    'Overall_R2': rf_overall_r2,    
    'Normal_R2': rf_n_r2,
    'Extreme_R2': rf_e_r2} 

logger.info('Random forest model done.')
``` 

The results from the Linear Regression and Random Forest models are combined into one pandas dataframe and saved as a parquet file. The results dataframe is also uploaded to duckdb to be used later for visualizations. 

```python
######## create results dataframe to compare models
results_df = pd.DataFrame([lr_results, rf_results])

# save results_df to parquet file for later use
results_df.to_parquet('data/model_results.parquet', engine='pyarrow', index=False)

######## save resultsdf to duckdb table 
con = None
try: 
    # create and verify connection 
    con = duckdb.connect(database='project1.db', read_only=False) 
    logger.info("Connected to duckdb instance.") 

    # inserting table 
    con.execute(f"""
        DROP TABLE IF EXISTS results_df;
        CREATE TABLE results_df AS
        SELECT * FROM read_parquet('data/model_results.parquet');
    """)

    logger.info("results_df loaded into duckdb table.")

except Exception as e:
    logger.error(f"An error occurred: {e}")

finally:
    if con:
        con.close()
        logger.info("Duckdb connection closed.")
```

## Visualization 
```python 
######## imports and logging 
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='visualization.log'
)
logger = logging.getLogger(__name__)


######## load duckdb table into pandas dfs 
con = None 
try: 
    # create and verify connection 
    con = duckdb.connect(database='project1.db', read_only=False) 
    logger.info("Connected to duckdb instance.") 

    # inserting tables 
    df = con.execute(f"""
        SELECT * FROM results_df;
    """).fetchdf()
    
    logger.info("results_df loaded into pandas dataframes")

except Exception as e:
    logger.error(f"An error occurred: {e}")

finally:
    if con:
        con.close()
        logger.info("Duckdb connection closed.")
```

The pandas.melt function is used here to transform the results dataframe into a more usable dataframe for making a bar chart, which is colored by model and separated by metric.

```python
######## transforming df into df2 for easier visualization 
df2 = df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Value")
``` 

The visualization was created with the goal of showing the difference in the models' performance in comparison to each other and to the three different scenarios. A bar chart was chosen because it allows for a clean visualization, while still showing all the necessary information and is easy to see the comparisons. There are two bar charts in the figure because of the difference in values between RMSE and MAE, and R^2. RMSE and MAE have similar value ranges that there is no issue plotting them on the same y-axis, however R^2 has a range of 0-1, which did not show well on the same plot as the other two metrics. They are shown in two separate graphs to increase readability and ensure that correct, and no misleading conclusions are drawn.

```python
######## create visualization 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# split data into rmse+mae and r2
errordf = df2[df2['Metric'].str.contains('MAE|RMSE')]
r2df = df2[df2['Metric'].str.contains('R2')]

# get metrics 
e_metrics = errordf['Metric'].unique()
r2_metrics = r2df['Metric'].unique()

# split by model 
lr_error_values = [errordf[(errordf["Model"]=="Linear_Regression") & (errordf["Metric"]==m)]["Value"].values[0] for m in e_metrics]
rf_error_values = [errordf[(errordf["Model"]=="Random_Forest") & (errordf["Metric"]==m)]["Value"].values[0] for m in e_metrics]

lr_r2_values = [r2df[(r2df["Model"]=="Linear_Regression") & (r2df["Metric"]==m)]["Value"].values[0] for m in r2_metrics]
rf_r2_values = [r2df[(r2df["Model"]=="Random_Forest") & (r2df["Metric"]==m)]["Value"].values[0] for m in r2_metrics]

# x positions 
x1 = np.arange(len(e_metrics))
x2 = np.arange(len(r2_metrics))
width = 0.35 

# plot rmse and mae
bars11 = ax1.bar(x1 - width/2, lr_error_values, width, label="Linear Regression", color='#1f85f2')
bars12 = ax1.bar(x1 + width/2, rf_error_values, width, label="Random Forest", color='#20a839')
ax1.bar_label(bars11, fmt='%.0f', padding=3)
ax1.bar_label(bars12, fmt='%.0f', padding=3)

ax1.set_xticks(x1)
# making x labels more readable
e_metrics = ['Overall\nRMSE', 'Normal\nRMSE', 'Extreme\nRMSE', 'Overall\nMAE', 'Normal\nMAE', 'Extreme\nMAE']
ax1.set_xticklabels(e_metrics, ha="center")
ax1.set_ylabel("Error Value")
ax1.set_xlabel('Metrics')

# plot r2 
bars21 = ax2.bar(x2 - width/2, lr_r2_values, width, label="Linear Regression", color='#1f85f2')
bars22 = ax2.bar(x2 + width/2, rf_r2_values, width, label="Random Forest", color='#20a839')
ax2.bar_label(bars21, fmt='%.3f', padding=3)
ax2.bar_label(bars22, fmt='%.3f', padding=3)

ax2.set_xticks(x2)
# making x labels more readable 
r2_metrics = ['Overall R^2', 'Normal R^2', 'Extreme R^2']
ax2.set_xticklabels(r2_metrics, ha="center")
ax2.set_ylabel("R^2 Value")
ax2.set_xlabel('Metrics')

# title, legend, caption
fig.suptitle("Model Performance Comparison using RMSE, MAE, and R^2 Metrics", fontsize=16, x=0.5, y=1.0)
plt.legend(loc='best', bbox_to_anchor=(0.17, 1.09), ncols=2)
fig.text(0.5, -0.09, 
         "This bar chart compares the performance of a Linear Regression model and a Random Forest model across three metrics: Root Mean Squared Error (RMSE), \nMean Absolute Error (MAE), and R-squared (R^2). Each plot shows the metrics for performance overall, performance on normal weather conditions, \nand performance on extreme weather conditions for each model.", 
         ha='center')

# save as png 
plt.savefig('visualization.png', bbox_inches='tight')
logger.info("Visualization created and saved as visualization.png")

plt.show() 

```

Using this visualization, we can see that the Random Forest model performs much better than the Linear Regression model in all three scenarios, across all metrics even though the RMSE and MAE values are high for both models. In terms of the models' prediction accuracy for extreme weather events compared to normal or overall weather events, there is an slight increase in error for the extreme weather events. However, looking at the R^2 values, for both models the R^2 values for extreme weather events is higher than the other two scenarios, which means that there was less variation in the model for these scenarios. Even though these models contained some error, this information can be useful to electricity companies in the future in terms of working to forecast the effects of extreme weather events on their energy demand. 