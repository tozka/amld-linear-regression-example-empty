# Electric Car Battery Linear Regression Example

## Linear Regression
Linear regression is often referred to as the building block of statistical learning methods. In a nutshell,
it is an attempt to model a relationship between two or more variables by fitting a linear equation
to the data at hand. For example, suppose you plot people's salaries on the y-axis and their years of
education on the x-axis in a simple scatter plot. In this sense, you are trying to estimate a dependent 
variable (salary) by using a predictor (years of education) by drawing a line of best fit through the data.
A line of best fit is a line that minimizes the sum of the distances from itself to each point in the data. The resulting
line's slope, is the coefficient of the predictor (i.e., what kind of effect one unit change of years of education
has on the predicted salary). For example, let's say that the line of best fit follows the equation below:
```
y = 30000 + 5000x1, where y = salary, x1 = years of education, and 30000 is a constant y intercept
```
This means that for one more year of education, a person's salary is estimated to increase by 5000, all else held constant.
Thus, for someone with 12 years of education, their predicted salary is:
```
y = 30000 + 5000*(12)
y = 90000
```
This is a rather simplistic example. In reality, we know there are many factors that influence a person's salary.
This is where multivariate linear regression comes into play. For example, suppose you now have information not only on
the person's salary and years of education, but their parents' last combined income, the years of work experience,
etc. You can estimate a model where each factor's effect is being considered, though visualizing the line of best fit
will get more difficult as you keep adding dimensions! Not to worry, the math still works! Here's an example of a
multivariate linear regression:
```
salary = 20000 + 4000*years_of_education + 1.1*last_combined_parents_income + 1000*years_of_experience
```
Thus, a person with 12 years of education, 100000 as their parents' last combined income and 5 years of experience
is estimated to earn:
```
y = 20000 + 4000*12 + 1.1*100000 + 1000*5
y = 183000
```
Linear regression contains a lot of aspects to it that need to be considered. Topics such as:
* How to estimate the coefficients
* The tradeoff between bias and variance
* Measuring the quality of fit and model accuracy
* Omitted variable bias
* Non-linear transformations of the predictors
* Interaction and dummy/binary variables

are only just a handful of topics that need to be considered. For a much better and a lot more detailed explanation of
linear regression (and statistical methods, in general) please visit: https://www.statlearning.com/.

## Electric Car Battery Example
### Data Source: 
https://www.kaggle.com/gktuzgl/id-3-pro-max-ev-consumption-data

### Explanation:
Using the data provided by Göktuğ Özgül on Kaggle.com, we will build a simple linear regression
model that predicts battery drainage: how much will your electric car's battery drain if you drive it 
in certain ways. For example, how much should you expect your battery to be drained if you drive 50 km at
50 km per hour, using heated seats?

We will cover:
* How to read in the data and deal with special characters
* How to explore the data, both for numerical variables and categorical variables
* How to process the data and very light feature engineering (i.e., creating new variables from existing ones)
* How to deal with multicollinearity
* One of the many possible ways to perform feature selection
* How to deal with possible data leakage
* How to build a simple linear regression model
* How to extract the results and predictions from the model
* How to build a simple Streamlit dashboard showcasing your model's predictive ability

### Sources Referred To:
* https://www.kaggle.com/gktuzgl/id-3-pro-max-ev-consumption-data
* https://www.statlearning.com/
* https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
* https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
* https://www.youtube.com/watch?v=Klqn--Mu2pE
* https://medium.com/codex/step-by-step-guide-to-simple-and-multiple-linear-regression-in-python-867ac9a30298

## Workshop Steps
Now that you have opened up the MyBinder environment and are reading this, you are already on the right track! Inside this environment,
you will also find:
* README.md: no need to worry about this for now
* requirements.txt: this is a list of required libraries that were installed upon startup
* setup.ipynb: This should open up on its own upon start up and you will need to add certain commands to make everything run
* VW ID. 3 Pro Max EV Consumption.csv: The raw .csv file from Kaggle.com. Please note that we will need to move this into the data job's folder once we
create it for a neater environment

### Step 1: Explore VDK's Functionalities
A simple command like that found in the setup.ipynb "!vdk --help" gives you all the information you need.

### Step 2: Create a Data Job
Now that we have explored VDK's capabilities, let's create our data job. Keep in mind that we would like to have a sub-folder for the data job,
so that our Streamlit script is outside of it and in the main directory. Based on the information above, try creating a data job titled "linear-reg-data-job".
You can chose any team name that you want, but please create the job at the home directory. This will create a sub-folder for the data job. The home
directory is /home/jovyan.

Hint: try typing "!vdk create --help" and play around with -n, -t, and -p. If you get really stuck, head all the way down for the answers.

### Step 3: Work Out the Data Job Template
Now that you have created a data job, please go inside the folder and set up the structure of your data job. Here's the idea:
* Let's have one Python script that reads in the data and strips its special characters and re-saves it
* Let's have another Python script that reads in the fixed data and performs exploratory data analysis
* Let's have a third Python script that reads in the data from the first script, cleans up the data, and gets it ready for model building and testing
* Lastly, let's have a Python script that reads the data from the third script, builds a simple Linear Regression model, tests it, and saves it

To do this, let's first get rid of the template SQL script. We won't be needing this for this example.

Let's rename the Python script as 10_read_in_data.py.

Now, let's also move the raw .CSV file from the main folder to the data job's sub-folder. 

At this point, you should have the following files inside the data job's folder:
* 10_read_in_data.py
* VW ID. 3 Pro Max EV Consumption.csv
* config.ini: This file keeps important information as to how frequently a data job should run, defines who is the team owning the data job,
as well as who is receiving emails for successful/ unsuccessful runs of your data job. This file is not needed for local runs.
* requirements.txt: A file containing a list of the libraries that your code will depend on to run.
* README.md: An optional file that you can delete, if you'd like. 

Please open up requirements.txt file inside the linear-reg-data-job sub-folder, delete everything, and paste the following code inside:
```
pyarrow==7.0.0
seaborn #seaborn==0.10.1
matplotlib
sklearn
xlsxwriter
```

Lastly, please go back to the setup.ipynb and open up a new code block. Inside it, please type and run:
```
!pip install -r "linear-reg-data-job/requirements.txt"
```
This command will install all of the libraries we will need to execute our data job sucessfully.


### Step 4: Data Job - Read in the Data and Strip Special Characters
Please open up 10_read_in_data.py. Inside it, you will see the code template already populated. You will also see a function called "run".
Above it, please delete everything and paste the following code, which imports some necessary libraries and initializes the log.

```
import pandas as pd
import os
import logging
import pathlib
from vdk.api.job_input import IJobInput

logger = logging.getLogger(__name__)
os.chdir(pathlib.Path(__file__).parent.absolute())
```

Inside the run function, please delete this part, so that the function's body is blank except for the docstrings:
```
log.info(f"Starting job step {__name__}")

# Write your python code inside here ... for example:
job_input.send_object_for_ingestion(
    payload=dict(id="Hello World!"), destination_table="hello_world"
```

Instead of it, please paste this bit of code, which will read in the data with the special characters, strip them, and re-save the file.
Please note, however, that we've left a bit of a challenge in the code: look for the [BLANK] in the code. What we want you to do there is to
apply a method that will finish off the cleaning of the column names by making them lowercase... hint: lower. 
```
logger.info('executing program: ' + os.path.basename(__file__))

# some definitions...
filename_to_import = 'VW ID. 3 Pro Max EV Consumption.csv'
filename_to_export = 'VW ID. 3 Pro Max EV Consumption_Fixed_Columns.csv'

# reading in the data with special characters...
df = pd.read_csv(
    filepath_or_buffer=filename_to_import,
    encoding='ISO-8859-1',
    parse_dates=['Date']
)

# stripping the non-alphanumeric characters in the column names and standardizing them...
df.columns = df.columns.\
    str.replace('[^a-zA-Z ]', '').\
    str.strip().\
    str.replace(' ', '_').\
    str.[BLANK]

# saving the dataset...
df.to_csv(path_or_buf=filename_to_export, index=False)
```

At this point, you have finished your first part of the data job. You have read in the data, stripped the special characters, and have
re-saved the data as a .CSV file with a new name. Good job!

If you're brave enough, you can go back to the setup.ipynb file and type "!vdk run linear-reg-data-job". What happens? Check out the log.
It is super helpful in debugging, if it gets to that. But hopefully, you will get a success message. You can open up the linear-reg-data-job
sub-folder and check that the new .CSV file is there. If so, let's move on to the second script!

### Step 5: Data Job - Exploratory Data Analysis
Let's go back to the linear-reg-data-job sub-folder and make a copy of the 10_read_in_data.py file. Let's call the copy 20_explore_data.py.
Open up the copy and let's make a few changes. 

Because we will be making some charts and tables, we will need to import some of the libraries that we installed from the requirements.txt file earlier.
As such, please delete all the code before the run function and paste the following:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pathlib
from vdk.api.job_input import IJobInput

logger = logging.getLogger(__name__)
os.chdir(pathlib.Path(__file__).parent.absolute())
```

Now, inside the run function, let's delete everything. We will write code that creates a sub-folder within the linear-reg-data-job sub-folder, which
will store all the exploratory graphics and tables. As such, delete everything and paste this:
```
logger.info('executing program: ' + os.path.basename(__file__))

if not os.path.exists('explore_data'):
    os.makedirs('explore_data')
```

Let's now read in the fixed .CSV file. Please paste the following code in the run function below the code that created the explore_data sub-folder.
Please note, however, that we've left the filename_to_import variable [BLANK]. Please change that to the name of the fixed .CSV file.
```
filename_to_import = [BLANK]
df = pd.read_csv(filepath_or_buffer=filename_to_import, parse_dates=['date'])
```

Let's also write some code below that to explore the data. Make sure to write it to the log using
```
logger.info(df.info())
```
as an example. Try logging the head and the tail of the data. Get creative with it!

At some point, however, we will need to define the numeric variables. Let's write the following code to store the column names of the numeric
variables.
```
num_cols = df.select_dtypes(include=np.number).columns.tolist()
```

Ok, now that we have the numeric columns. Let's create some histograms for each numeric column. Here is one way to do it using seaborn and matplotlib.
Try out some different functions. Try changing distplot to something else.
```
sns.set(
    style='ticks',
    rc={
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.figsize': (18, 14)
        }
    )
for num_col in num_cols:
    sns.distplot(df[num_col],
        bins=100).set(
            xlabel=num_col,
            ylabel='Count'
            )
    plt.savefig('explore_data/' + num_col + '.png')
    # plt.show()
    plt.clf()
```

Ok, now that we have created a histogram for each numeric variable and saved that histogram in the explore_data subfolder, let's turn our attention
to the categorical variables. Let's follow the same process as that above. In other words, let's first create a list of the names of the columns
that are categorical! We've left a little challenge for you in it: change [BLANK] to something that works!
```
cat_cols = [i for i in df.columns if i not in [BLANK]]
```

Let's now run a loop that calculates the value counts for each categorical variable. What does that mean? We want to see the values that occur
for each categorical variable and how often those values occur. We also want to save the result in an Excel file, where each worksheet within the
Excel file is a value count for each categorical variable. Here's the code to do that:

```
cat_writer = pd.ExcelWriter('explore_data/explore_categoricals.xlsx', engine='xlsxwriter')
for cat_col in cat_cols:
    temp = pd.DataFrame(
        df[cat_col].value_counts(dropna=False)
    )
    temp.to_excel(cat_writer, sheet_name=cat_col)
cat_writer.save()
```

After all this, let's go and run the data job again by re-running the "!vdk run..." command in the setup.ipynb.
This time, notice that it will run both scripts, one after the other in an alphabetical order. Let's go and check out the 
results! Please head over to the explore_data sub-folder within the linear-reg-data-job folder!
Congrats!

### Step 6: Data Job - Processing The Data
Now that we have explored the data, we know what we want to do and what we must do. Let's head over to the linear-reg-data-job sub-folder and create
a copy of the last Python script. Rename it 30_process_data.py.

Let's delete everything above the run function, import our libraries and initialize the log:
```
import pandas as pd
import numpy as np
import logging
import pathlib
import os
from vdk.api.job_input import IJobInput

logger = logging.getLogger(__name__)
os.chdir(pathlib.Path(__file__).parent.absolute())
```

Now, inside the run function, delete everything. Let's define the data sets that we will be reading in and later
saving and read in the data:
```
logger.info('executing program: ' + os.path.basename(__file__))

# some definitions...
filename_to_import = 'VW ID. 3 Pro Max EV Consumption_Fixed_Columns.csv'
filename_to_export = 'VW ID. 3 Pro Max EV Consumption_Model_Data.csv'

# reading in the data...
df = pd.read_csv(filepath_or_buffer=filename_to_import, parse_dates=['date'])
```

If you remember from the exploratory data analysis section, we had some missing values in our data set. There many ways to deal with missing values
but the simplest is to just drop observations where any column has a missing value. In our case, that doesn't drop too many data points, so let's go
ahead and do that:
```
df_no_nulls = df.copy().dropna()
```

Linear regression only works with numeric variables. As such, if we want to use the categorical variables, we will need to turn them into numerics:
```
df_no_nulls['ac_use'] = np.where(df_no_nulls['ac_c'] == 'OFF', 0, 1)

df_no_nulls['heated_seats'] = np.where(df_no_nulls['heated_front_seats_level'] == 0, 0, 1)

df_no_nulls['eco_mode'] = np.where(df_no_nulls['mode'] == 'ECO', 1, 0)

df_no_nulls["is_summer"] = np.where((df_no_nulls['date'] >= '2021-06-21') & (df_no_nulls['date'] <= '2021-09-22'), 1, 0)

df_no_nulls['is_bridgestone_tyre'] = np.where(df_no_nulls['tyres'] == 'Bridgestone 215/45 R20 LM32', 1, 0)
```

Let's now create our dependent variable - the variable we will be trying to estimate: battery drainage:
```
df_no_nulls['battery_drain'] = -(df_no_nulls['charge_level_end'] - df_no_nulls['charge_level_start'])
```

Lastly, let's also create one more variable: temperature change. Who knows - it may have explanatory power!
```
df_no_nulls['temperature_increase'] = df_no_nulls['temperature_end_c'] - df_no_nulls['temperature_start_c']
```

Let's now limit the data set to the variables we want. This helps declutter.
```
df_no_nulls_limited = df_no_nulls.copy()[
    [
        'battery_drain',
        'charge_level_start',
        'is_bridgestone_tyre',
        'temperature_start_c',
        'temperature_increase',
        'distance_km',
        'average_speed_kmh',
        'average_consumption_kwhkm',
        'ac_use',
        'heated_seats',
        'eco_mode',
        'is_summer'
    ]
]
```

It is always good practice, however, to look at the variables you created. Who knows - maybe you didn't see something with regard to the relationship
between some of the variables. For example, temperature start and temperature end might look perfectly fine on their own, but if we calculate the change
we may find some data entry error if we see that the temperature change was 40 degrees Celsius, for example. Let's look at the data.
```
logger.info(df_no_nulls_limited.describe())
logger.info(df_no_nulls_limited.loc[df_no_nulls_limited['battery_drain'] < 0])
```

We are definitely seeing one weird result: a negative battery drainage. That can't be. Let's remove the data point and continue:
```
df_no_nulls_limited_final = df_no_nulls_limited.copy().loc[
        df_no_nulls_limited['battery_drain'] >= 0]
logger.info(df_no_nulls_limited_final.describe()) 

df_no_nulls_limited_final.to_csv(
    path_or_buf=filename_to_export,
    index=False
```

Awesome! We have now processed the data and it's ready to be modeled. Let's run the job again, just to make sure that all of
the scripts function as they should! Remember, just go back to setup.ipynb and re-run the "!vdk run..." command.

### Step 7: Data Job - Build the Model, Test the Model, Save the Model
Ok, so let's create a copy of 30_process_data.py and name it 40_build_model.py. Open it up.

Let's delete everything above the run function again and paste the following:
```

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import pickle
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vdk.api.job_input import IJobInput

logger = logging.getLogger(__name__)
os.chdir(pathlib.Path(__file__).parent.absolute())
```

Let's now turn our attention to the run function. Let's get rid of everything in there and begin building our code.
First, we're going to want to set the file name to import, create a sub-folder to house our model, and read in the data.
```
logger.info('executing program: ' + os.path.basename(__file__))

filename_to_import = 'VW ID. 3 Pro Max EV Consumption_Model_Data.csv'

if not os.path.exists('model'):
    os.makedirs('model')

df = pd.read_csv(filepath_or_buffer=filename_to_import)
```

Let's now split the data into four chunks: x_train, y_train, x_test, and y_test. x denotes the independent variables or the
predictor variables and y denotes the variable we are trying to estimate - i.e., battery drainage. What's the idea here?
Well, suppose that you build a really good model. Well, how do you know it's good? By setting aside some testing data,
you can create a model purely on the training data and THEN test that model on data it has not seen before. That way,
you'll know how good the model is because you actually have the testing data's dependent variable (y test) and can measure it
against your model's prediction. The code below shows one way to split the data, but there are many, and we encourage 
you to read up more about this. We will split the data based on a pre-defined random state, so that the numbers are
reproducible. We will take 20 percent of the data and put it aside as testing data:
```
y = df.copy()[['battery_drain']]
x = df.copy().drop('battery_drain', axis=1)
x_train, x_test, y_train, y_test = train_test_split(
   x,
   y,
   test_size=0.2,
   random_state=22
 )
```

Let's check out what this looks like:
```
data_sets = {
    "x_train": x_train,
    "x_test": x_test,
    "y_train": y_train,
    "y_test": y_test
}
for name, data_set in data_sets.items():
    logger.info(f"The shape of the {name} dataset is: {data_set.shape}")
```

We will want to check that none of the independent variables (or predictors) are not heavily correlated with one another.
We can do this through a correlation plot:
```
sns.set(
    style='ticks',
    rc={'figure.figsize': (18, 14)}
)
sns.heatmap(
    x_train.corr(),
    annot=True,
    cmap=sns.diverging_palette(10, 250, n=240)
)
plt.savefig('explore_data/features_correlation.png')
# plt.show() the following features need to be dropped: is_bridgestone_tyre, is_summer, and ac_use
# plt.clf()
```

For those that appear to be heavily correlated with other main variables, we can drop them from both the train and test:
```
predictive_data_sets = [x_train, x_test]
for data_set in predictive_data_sets:
    data_set.drop(['is_bridgestone_tyre', 'is_summer', 'ac_use'], axis=1, inplace=True)
```

It is often good practice to perform feature selection. That means to narrow down the features/predictors you want to use to
be in your model. There are many ways to do this, but one is called Lasso regression. It penalizes the coefficients of the
least important predictors in your model and brings them to 0. Thus, we can manually take them out. Please make sure to
normalize the data first! We've left a little [BLANK] in there for you!
```
lasso = LassoCV(normalize=[BLANK], random_state=22)
lasso.fit(
    x_train,
    y_train
)
lasso = pd.Series(lasso.coef_, index=x_train.columns)

features_to_delete = list(lasso[lasso == 0].index)
for data_set in predictive_data_sets:
    data_set.drop(features_to_delete, axis=1, inplace=True)
```

OK, let's finally fit our model to the training data! We can then create a prediction based on the predictors' values
from x_test. Then, we can compare how those predictions compare against y_test. Neat, huh?!
```
linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_pred = linreg.predict(x_test)
y_pred = pd.DataFrame(y_pred, columns=['battery_drain_prediction'])
actual_vs_predicted = pd.concat(
    [y_test.copy().reset_index(drop=True), y_pred.copy().reset_index(drop=True)],
    axis=1
)
actual_vs_predicted.to_csv('model/actual_vs_model_predicted_battery_drain_test.csv')
```

Now that we have the true values of battery drain from y_test and the predicted battery drain from y_pred, we can get
some measures of model quality, like the mean squared error, mean absolute error, and r squared!
```
measurements = {
    'mean squared error': mean_squared_error,
    'mean absolute error': mean_absolute_error,
    'R2': r2_score}
for measure, func in measurements.items():
    logger.info(f"The {measure} is: {func(y_pred, y_test)}")
```

Let's also extract the coefficients of the model - who knows! Maybe we will need them!
```
coeff = pd.DataFrame(linreg.coef_).transpose()
inter = pd.DataFrame(linreg.intercept_).transpose()
inter_and_coeff = pd.concat(
    [inter, coeff],
    ignore_index=True
)
inter_and_coeff.columns = ['coefficients']
intercept = ['intercept']
intercept.extend(x_train.columns.to_list())
feature_names = pd.DataFrame(
    intercept,
    columns=['feature']
)
model_coeffs = pd.concat(
    [inter_and_coeff, feature_names],
    axis=1,
    join='outer'
)
model_coeffs.to_csv('model/model_coefficients.csv')
```

Finally, let's save the model!
```
filename = 'model/amld_linear_regression_model.sav'
pickle.dump(linreg, open(filename, 'wb'))
```

Let's run the data job again!
If it all goes through, congratulations! You have built and ran your first data job. No better feeling!

### Step 8: Let's Build a Streamlit Visualization!
Now that we have finished with the data job, let's use that hard-earned model to make a cool dashboard!

First, let's take a look at the coefficients from our model and let's build a parameters file that will serve as the
user input generated values for our predictors. They will be fed to the model, and we'll have a prediction for whatever
values a user gives us for the predictors! Neat!

Open up the linear-reg-data-job sub-folder and find the model_coefficients .CSV file. Take a note of the feature
column, as that will show you which predictors made it to the model and which predictors we will need to add in the
parameters.py file. As such, let's go back to the main folder and create a new script. Title it 'parameters.py' and
input the following code inside it:
```
parameters = {
    'charge_level_start': dict(
        label="Select Value for the Starting Charge Level",
        value=80,
        max_value=100,
        min_value=1
    ),
    'temperature_start_c': dict(
        label="What's the Temperature Outside? In Celsius Please!",
        value=20,
        max_value=50,
        min_value=-50
    ),
    'distance_km': dict(
        label="How Many Kilometers Are We Going to Drive?",
        value=50,
        max_value=1000,
        min_value=1
    ),
    'average_speed_kmh': dict(
        label="What's the Average Speed We Expect (in Kilometers per Hour)?",
        value=60,
        max_value=300,
        min_value=1
    ),
    'average_consumption_kwhkm': dict(
        label="Any Guesses on the Average Consumption (in kwhkm)?",
        value=15,
        max_value=50,
        min_value=1
    ),
    'heated_seats': dict(
        label="Do We Plan on Using the Heated Seats? (1 = Yes, 0 = No)",
        value=1,
        min_value=0,
        max_value=1
    ),
    'eco_mode': dict(
        label="Do We Plan on Using the Eco Mode? (1 = Yes, 0 = No)",
        value=1,
        min_value=0,
        max_value=1
    )
}
```
This will help set up the user input interface and will bound the user to a min and max value for the predictors.
In other words, a user won't be able to say that they expect to drive a 10000 km per hour.

Close out of the file and create a new Python script in the main folder. This will be the main script for the Streamlit
visualization. Title it "build_streamlit_dashboard.py".

Inside it, let's start with importing the libraries and adding titles:
```
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pathlib
import streamlit as st
from parameters import parameters

st.title('Electric Cars and Battery Drain Linear Regression Example')
```

Now, let's do the first section of our visualization: the model showcase and quality measure:
```
os.chdir(pathlib.Path(__file__).parent.absolute())
actual_vs_pred_loc = 'linear-reg-data-job/model/actual_vs_model_predicted_battery_drain_test.csv'
model_loc = 'linear-reg-data-job/model/amld_linear_regression_model.sav'
```

Now, we'll read in the actual versus predicted data set that we created in our data job and visualize it in the dashboard:
```
actual_vs_pred = pd.read_csv(actual_vs_pred_loc, usecols=range(1, 3))
actual_vs_pred['absolute_difference'] = abs(
    actual_vs_pred['battery_drain'] - actual_vs_pred['battery_drain_prediction']
)
st.dataframe(actual_vs_pred)
battery_drain = actual_vs_pred[['battery_drain']]
battery_drain_prediction = actual_vs_pred[['battery_drain_prediction']]
```

Let's now visualize the model's quality metrics!
```
mse = round(mean_squared_error(battery_drain, battery_drain_prediction), 2)
mae = round(mean_absolute_error(battery_drain, battery_drain_prediction), 2)
r2  = round(r2_score(battery_drain, battery_drain_prediction), 2)

st.metric("The Mean Squared Error of This Model On This Testing Data Is:", mse)
st.metric("The Mean Absolute Error of This Model On This Testing Data Is:", mae)
st.metric("The R2 is:", r2)
```

Now let's add the second section, which will allow the user to select the predictors' values that will feed into the
model and give them an estimated battery drain!
```
st.header('How Much Will Your Electric Car Battery Drain? You May Be Surprised!')
st.write("Enter Your Custom Values in the SideBar - Please Enter Sensible Values Only!")


results = {}
for measurement, params in parameters.items():
    output = st.sidebar.number_input(**params)
    results[measurement] = output
results_df = pd.DataFrame(results, index=[0])


pickled_model = pickle.load(open(model_loc, 'rb'))


estimate = pickled_model.predict(results_df)
```

Let's fix some minor stuff like: if the user enters inputs that generate a drain prediction higher than the initial charge,
let's tell them that they will run out of battery! Or if the model somehow predicts a negative charge, to fix that to 0.
I know, an easy way out! :)
```
if estimate > results['charge_level_start']:
    estimate = results['charge_level_start']
    st.metric("Your Estimated Battery Drainage (in Percent) Is:", estimate)
    st.write("Note: The Model's Estimate Exceeds the Starting Level Charge; Thus Estimate is Capped")
elif estimate < 0:
    estimate = 0
    st.metric("Your Estimated Battery Drainage (in Percent) Is:", estimate)
else:
    st.metric("Your Estimated Battery Drainage (in Percent) Is:", estimate)
```

Congratulations! You have built your first Streamlit dashboard that even allows for a user to enter inputs! How cool is that?

As a last step, go back to the setup.ipynb and type the following code:
```
!streamlit run building_streamlit_dashboard.py
```

You will get an output, but the kernel will be stuck. That's okay! Just open a new tab in your browser,
copy the link of the MyBinder environment, delete everything after "user/blah blah blah" and paste "/proxy/8501/"
So, something like this: 
```
https://hub.gke2.mybinder.org/user/alexanderavramo-n-example-empty-zkd8q00p/proxy/8501/
```

The Streamlit dashboard will now show up!


## Workshop Answers
### Step 1: Explore VDK's Functionalities
```
!vdk --help
```
### Step 2: Create a Data Job
```
!vdk create --help
!vdk create -n linear-reg-data-job -t team-awesome -p /home/jovyan
```
### Step 3: Work Out the Data Job Template
Inside the linear-reg-data-job subfolder:
* Get rid of the template SQL script
* Rename the Python script as 10_read_in_data.py
* Move the raw .CSV file from the main folder to the data job's sub-folder
* Open up requirements.txt file inside the linear-reg-data-job sub-folder, delete everything, and paste the following code inside:
```
pyarrow==7.0.0
seaborn #seaborn==0.10.1
matplotlib
sklearn
xlsxwriter
```
* Lastly, please go back to the setup.ipynb and open up a new code block. Inside it, please type and run:
```
!pip install -r "linear-reg-data-job/requirements.txt"
```

### Step 4: Data Job - Read in the Data and Strip Special Characters
```
import pandas as pd
import os
import logging
import pathlib
from vdk.api.job_input import IJobInput

logger = logging.getLogger(__name__)
os.chdir(pathlib.Path(__file__).parent.absolute())

def run(job_input: IJobInput):
    logger.info('executing program: ' + os.path.basename(__file__))

    # some definitions...
    filename_to_import = 'VW ID. 3 Pro Max EV Consumption.csv'
    filename_to_export = 'VW ID. 3 Pro Max EV Consumption_Fixed_Columns.csv'

    # reading in the data with special characters...
    df = pd.read_csv(
        filepath_or_buffer=filename_to_import,
        encoding='ISO-8859-1',
        parse_dates=['Date']
    )

    # stripping the non-alphanumeric characters in the column names and standardizing them...
    df.columns = df.columns.\
        str.replace('[^a-zA-Z ]', '').\
        str.strip().\
        str.replace(' ', '_').\
        str.lower()

    # saving the dataset...
    df.to_csv(path_or_buf=filename_to_export, index=False)
```
```
!vdk run linear-reg-data-job
```

### Step 5: Data Job - Exploratory Data Analysis
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pathlib
from vdk.api.job_input import IJobInput

logger = logging.getLogger(__name__)
os.chdir(pathlib.Path(__file__).parent.absolute())

def run(job_input: IJobInput):
    logger.info('executing program: ' + os.path.basename(__file__))

    # creating a sub-folder within our data job folder to store the exploratory graphics and tables...
    if not os.path.exists('explore_data'):
        os.makedirs('explore_data')

    # some definitions...
    filename_to_import = 'VW ID. 3 Pro Max EV Consumption_Fixed_Columns.csv'

    # reading in the data with fixed column names, as outputted from 10_read_in_data.py...
    df = pd.read_csv(filepath_or_buffer=filename_to_import, parse_dates=['date'])

    # exploring the structure of the data...
    logger.info('Information about the data: ')
    logger.info(df.info())
    pd.set_option('display.max.columns', None)
    logger.info(df.head())
    logger.info(df.tail())

    # exploring the numeric variables through plotting histograms...
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    sns.set(
        style='ticks',
        rc={
            'axes.spines.right': False,
            'axes.spines.top': False,
            'figure.figsize': (18, 14)
            }
        )
    for num_col in num_cols:
        sns.distplot(
            df[num_col],
            bins=100).\
            set(xlabel=num_col,
                ylabel='Count'
                )
        plt.savefig('explore_data/' + num_col + '.png')
        # plt.show()
        plt.clf()

    # exploring the categorical variables through examining the commonly occurring values...
    cat_cols = [i for i in df.columns if i not in num_cols]
    cat_writer = pd.ExcelWriter('explore_data/explore_categoricals.xlsx', engine='xlsxwriter')
    for cat_col in cat_cols:
        temp = pd.DataFrame(
            df[cat_col].value_counts(dropna=False)
        )
        temp.to_excel(cat_writer, sheet_name=cat_col)
    cat_writer.save()
```

### Step 6: Data Job - Processing The Data
```
import pandas as pd
import numpy as np
import logging
import pathlib
import os
from vdk.api.job_input import IJobInput

logger = logging.getLogger(__name__)
os.chdir(pathlib.Path(__file__).parent.absolute())

def run(job_input: IJobInput):
    logger.info('executing program: ' + os.path.basename(__file__))

    # some definitions...
    filename_to_import = 'VW ID. 3 Pro Max EV Consumption_Fixed_Columns.csv'
    filename_to_export = 'VW ID. 3 Pro Max EV Consumption_Model_Data.csv'

    # reading in the data...
    df = pd.read_csv(filepath_or_buffer=filename_to_import, parse_dates=['date'])

    # dropping missing values...
    df_no_nulls = df.copy().dropna()

    # encoding the categorical variables....
    df_no_nulls['ac_use'] = np.where(
        df_no_nulls['ac_c'] == 'OFF', 0, 1
    )
    df_no_nulls['heated_seats'] = np.where(
        df_no_nulls['heated_front_seats_level'] == 0, 0, 1
    )
    df_no_nulls['eco_mode'] = np.where(
        df_no_nulls['mode'] == 'ECO', 1, 0
    )
    df_no_nulls["is_summer"] = np.where(
        (df_no_nulls['date'] >= '2021-06-21') & (df_no_nulls['date'] <= '2021-09-22'), 1, 0
    )
    df_no_nulls['is_bridgestone_tyre'] = np.where(
        df_no_nulls['tyres'] == 'Bridgestone 215/45 R20 LM32', 1, 0
    )

    # creating additional variables...
    df_no_nulls['battery_drain'] = \
        -(df_no_nulls['charge_level_end'] - df_no_nulls['charge_level_start'])
    df_no_nulls['temperature_increase'] = \
        df_no_nulls['temperature_end_c'] - df_no_nulls['temperature_start_c']

    # clearing the dataset of all the clutter...
    df_no_nulls_limited = df_no_nulls.copy()[
        [
            'battery_drain',
            'charge_level_start',
            'is_bridgestone_tyre',
            'temperature_start_c',
            'temperature_increase',
            'distance_km',
            'average_speed_kmh',
            'average_consumption_kwhkm',
            'ac_use',
            'heated_seats',
            'eco_mode',
            'is_summer'
        ]
    ]

    # observing the processed data and making corrections, as needed...
    logger.info(df_no_nulls_limited.describe())  # we see a possible data error
    logger.info(df_no_nulls_limited.loc[df_no_nulls_limited['battery_drain'] < 0])  # this is an error we have to remove
    df_no_nulls_limited_final = df_no_nulls_limited.copy().loc[
        df_no_nulls_limited['battery_drain'] >= 0]
    logger.info(df_no_nulls_limited_final.describe())  # looks good now

    # saving the processed data...
    df_no_nulls_limited_final.to_csv(
        path_or_buf=filename_to_export,
        index=False
    )
```

### Step 7: Data Job - Build the Model, Test the Model, Save the Model
```
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import pickle
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vdk.api.job_input import IJobInput

logger = logging.getLogger(__name__)
os.chdir(pathlib.Path(__file__).parent.absolute())

def run(job_input: IJobInput):
    logger.info('executing program: ' + os.path.basename(__file__))

    # some definitions...
    filename_to_import = 'VW ID. 3 Pro Max EV Consumption_Model_Data.csv'

    # creating a sub-folder to house the model and model related things...
    if not os.path.exists('model'):
        os.makedirs('model')

    # reading in the data created from our processing program...
    df = pd.read_csv(filepath_or_buffer=filename_to_import)

    # splitting the data into a training dataset and a testing dataset...
    y = df.copy()[['battery_drain']]
    x = df.copy().drop('battery_drain', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
       x,
       y,
       test_size=0.2,
       random_state=22
     )

    # explore the split data
    data_sets = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test
    }
    for name, data_set in data_sets.items():
        logger.info(f"The shape of the {name} dataset is: {data_set.shape}")

    # checking for multicollinearity...
    sns.set(
        style='ticks',
        rc={'figure.figsize': (18, 14)}
    )
    sns.heatmap(
        x_train.corr(),
        annot=True,
        cmap=sns.diverging_palette(10, 250, n=240)
    )
    plt.savefig('explore_data/features_correlation.png')
    # plt.show() the following features need to be dropped: is_bridgestone_tyre, is_summer, and ac_use
    # plt.clf()

    # drop the heavily correlated variables from both the training and testing data sets...
    predictive_data_sets = [x_train, x_test]
    for data_set in predictive_data_sets:
        data_set.drop(['is_bridgestone_tyre', 'is_summer', 'ac_use'], axis=1, inplace=True)

    # using Lasso regularization to delete less important features...
    lasso = LassoCV(normalize=True, random_state=22)
    lasso.fit(
        x_train,
        y_train
    )
    lasso = pd.Series(lasso.coef_, index=x_train.columns)

    # deleting the less important features from the train and test data sets...
    features_to_delete = list(lasso[lasso == 0].index)
    for data_set in predictive_data_sets:
        data_set.drop(features_to_delete, axis=1, inplace=True)

    # fitting the model on the training data...
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)

    # testing the model on testing data and extracting predictions...
    y_pred = linreg.predict(x_test)
    y_pred = pd.DataFrame(y_pred, columns=['battery_drain_prediction'])
    actual_vs_predicted = pd.concat(
        [y_test.copy().reset_index(drop=True), y_pred.copy().reset_index(drop=True)],
        axis=1
    )
    actual_vs_predicted.to_csv('model/actual_vs_model_predicted_battery_drain_test.csv')

    # obtaining model accuracy...
    measurements = {
        'mean squared error': mean_squared_error,
        'mean absolute error': mean_absolute_error,
        'R2': r2_score}
    for measure, func in measurements.items():
        logger.info(f"The {measure} is: {func(y_pred, y_test)}")

    # extracting the coefficients...
    coeff = pd.DataFrame(linreg.coef_).transpose()
    inter = pd.DataFrame(linreg.intercept_).transpose()
    inter_and_coeff = pd.concat(
        [inter, coeff],
        ignore_index=True
    )
    inter_and_coeff.columns = ['coefficients']
    intercept = ['intercept']
    intercept.extend(x_train.columns.to_list())
    feature_names = pd.DataFrame(
        intercept,
        columns=['feature']
    )
    model_coeffs = pd.concat(
        [inter_and_coeff, feature_names],
        axis=1,
        join='outer'
    )
    model_coeffs.to_csv('model/model_coefficients.csv')

    # saving the model...
    filename = 'model/amld_linear_regression_model.sav'
    pickle.dump(linreg, open(filename, 'wb'))
```
### Step 8: Let's Build a Streamlit Visualization!
Create the parameters.py in the main folder. Inside it, paste:
```
parameters = {
    'charge_level_start': dict(
        label="Select Value for the Starting Charge Level",
        value=80,
        max_value=100,
        min_value=1
    ),
    'temperature_start_c': dict(
        label="What's the Temperature Outside? In Celsius Please!",
        value=20,
        max_value=50,
        min_value=-50
    ),
    'distance_km': dict(
        label="How Many Kilometers Are We Going to Drive?",
        value=50,
        max_value=1000,
        min_value=1
    ),
    'average_speed_kmh': dict(
        label="What's the Average Speed We Expect (in Kilometers per Hour)?",
        value=60,
        max_value=300,
        min_value=1
    ),
    'average_consumption_kwhkm': dict(
        label="Any Guesses on the Average Consumption (in kwhkm)?",
        value=15,
        max_value=50,
        min_value=1
    ),
    'heated_seats': dict(
        label="Do We Plan on Using the Heated Seats? (1 = Yes, 0 = No)",
        value=1,
        min_value=0,
        max_value=1
    ),
    'eco_mode': dict(
        label="Do We Plan on Using the Eco Mode? (1 = Yes, 0 = No)",
        value=1,
        min_value=0,
        max_value=1
    )
}
```
Inside the main folder, create "build_streamlit_dashboard.py". Inside it, paste:
```
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pathlib
import streamlit as st
from parameters import parameters

# setting up the title of the page...
st.title('Electric Cars and Battery Drain Linear Regression Example')

# setting up the sub-header for the first part of the page...
st.header("Our Simple Model's Accuracy")
st.write("How Did Our Model Perform?")

# some definitions...
os.chdir(pathlib.Path(__file__).parent.absolute())
actual_vs_pred_loc = 'linear-reg-data-job/model/actual_vs_model_predicted_battery_drain_test.csv'
model_loc = 'linear-reg-data-job/model/amld_linear_regression_model.sav'

# reading in the actual versus predicted data set...
actual_vs_pred = pd.read_csv(actual_vs_pred_loc, usecols=range(1, 3))
actual_vs_pred['absolute_difference'] = abs(
    actual_vs_pred['battery_drain'] - actual_vs_pred['battery_drain_prediction']
)
st.dataframe(actual_vs_pred)
battery_drain = actual_vs_pred[['battery_drain']]
battery_drain_prediction = actual_vs_pred[['battery_drain_prediction']]

# outputting some performance metrics...
mse = round(mean_squared_error(battery_drain, battery_drain_prediction), 2)
mae = round(mean_absolute_error(battery_drain, battery_drain_prediction), 2)
r2  = round(r2_score(battery_drain, battery_drain_prediction), 2)

st.metric("The Mean Squared Error of This Model On This Testing Data Is:", mse)
st.metric("The Mean Absolute Error of This Model On This Testing Data Is:", mae)
st.metric("The R2 is:", r2)

# setting up the sub-headers for the second part of the page...
st.header('How Much Will Your Electric Car Battery Drain? You May Be Surprised!')
st.write("Enter Your Custom Values in the SideBar - Please Enter Sensible Values Only!")

# selecting the user inputs...
results = {}
for measurement, params in parameters.items():
    output = st.sidebar.number_input(**params)
    results[measurement] = output
results_df = pd.DataFrame(results, index=[0])

# reading in the model...
pickled_model = pickle.load(open(model_loc, 'rb'))

# obtaining model prediction...
estimate = pickled_model.predict(results_df)

# printing model prediction...
if estimate > results['charge_level_start']:
    estimate = results['charge_level_start']
    st.metric("Your Estimated Battery Drainage (in Percent) Is:", estimate)
    st.write("Note: The Model's Estimate Exceeds the Starting Level Charge; Thus Estimate is Capped")
elif estimate < 0:
    estimate = 0
    st.metric("Your Estimated Battery Drainage (in Percent) Is:", estimate)
else:
    st.metric("Your Estimated Battery Drainage (in Percent) Is:", estimate)
```

Go back to setup.ipynb and type:
```
!streamlit run build_streamlit_dashboard.py
```

Copy the current URL, create a new tab, paste the URL, delete everything after "/user/blahblahlbah" and paste "/proxy/8501/"
So, something like this: 
```
https://hub.gke2.mybinder.org/user/alexanderavramo-n-example-empty-zkd8q00p/proxy/8501/
```

## VDK
Versatile Data Kit feature allows you to implement automated pull ingestion and batch data processing.

## Create the data job Files

Data Job directory can contain any files, however there are some files that are treated in a specific way:

* SQL files (.sql) - called SQL steps - are directly executed as queries against your configured database;
* Python files (.py) - called Python steps - are Python scripts that define run function that takes as argument the job_input object;
* config.ini is needed in order to configure the Job. This is the only file required to deploy a Data Job;
* requirements.txt is an optional file needed when your Python steps use external python libraries.

Delete all files you do not need and replace them with your own.

## Data Job Code

VDK supports having many Python and/or SQL steps in a single Data Job. Steps are executed in ascending alphabetical order based on file names.
Prefixing file names with numbers makes it easy to have meaningful file names while maintaining the steps' execution order.

Run the Data Job from a Terminal:
* Make sure you have vdk installed. See Platform documentation on how to install it.
```
vdk run <path to Data Job directory>
```

## Deploy Data Job

When a Job is ready to be deployed in a Versatile Data Kit runtime (cloud):
Run the command below and follow its instructions (you can see its options with `vdk --help`)
```python
vdk deploy
```