import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

# Load a matrix from file
data = pd.read_csv('ENB2012_data.csv')

# Normalize the data to avoid numerical stability issues
normalized_data = (data - data.mean())/data.std()

data.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area',
                'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']

X = data.iloc[:,0:8]
y1 = data.iloc[:,8:9]   #Heating Load
y2 = data.iloc[:,9:10]  #Cooling Load

# First, we create our training and test set from the data

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# create a copy of the training set for reference

train_copy = train_set.copy()

# Separate the inputs from the outputs
X_t = train_copy.drop("cooling_load", axis=1)
X_train = X_t.drop("heating_load", axis=1)
y_train1 = train_copy["heating_load"].copy()
y_train2 = train_copy["cooling_load"].copy()


# Description of data
def data_description():
    print("Data shape = ", data.shape)
    print("\nData description", data.describe())
    print("\n\nCorrelation", data.corr())

# Data Analysis and Visualization

# We try to explore more, some statistical properties of the data, we plot the probability densities
# to obtain the probability density estimates using histograms
def visualization():
    plt.hist(X.relative_compactness)
    plt.ylabel('frequency')
    plt.xlabel('X1')
    plt.savefig('r_c.png')
    plt.show()
    plt.hist(X.surface_area)
    plt.ylabel('frequency')
    plt.xlabel('X2')
    plt.savefig('s_a.png')
    plt.show()
    plt.hist(X.wall_area)
    plt.ylabel('frequency')
    plt.xlabel('X3')
    plt.savefig('w_a.png')
    plt.show()
    plt.hist(X.roof_area)
    plt.ylabel('frequency')
    plt.xlabel('X4')
    plt.savefig('r_a.png')
    plt.show()
    plt.hist(X.overall_height)
    plt.ylabel('frequency')
    plt.xlabel('X5')
    plt.savefig('o_h.png')
    plt.show()
    plt.hist(X.orientation)
    plt.ylabel('frequency')
    plt.xlabel('X6')
    plt.savefig('orien.png')
    plt.show()
    plt.hist(X.glazing_area)
    plt.ylabel('frequency')
    plt.xlabel('X7')
    plt.savefig('g_a.png')
    plt.show()
    plt.hist(X.glazing_area_distribution)
    plt.ylabel('frequency')
    plt.xlabel('X8')
    plt.savefig('g_a_d.png')
    plt.show()
    plt.hist(y1.heating_load)
    plt.ylabel('frequency')
    plt.xlabel('y1')
    plt.savefig('h_l.png')
    plt.show()
    plt.hist(y2.cooling_load)
    plt.ylabel('frequency')
    plt.xlabel('y2')
    plt.savefig('c_l.png')
    plt.show()

    # visualize the relationship between the features and the response using scatterplots
    fig, axs = plt.subplots(1, 8, sharey=True)
    # For heating load
    normalized_data.plot(kind='scatter', x='X1', y='Y1', ax=axs[0], figsize=(16, 8))
    normalized_data.plot(kind='scatter', x='X2', y='Y1', ax=axs[1])
    normalized_data.plot(kind='scatter', x='X3', y='Y1', ax=axs[2])
    normalized_data.plot(kind='scatter', x='X4', y='Y1', ax=axs[3])
    normalized_data.plot(kind='scatter', x='X5', y='Y1', ax=axs[4])
    normalized_data.plot(kind='scatter', x='X6', y='Y1', ax=axs[5])
    normalized_data.plot(kind='scatter', x='X7', y='Y1', ax=axs[6])
    normalized_data.plot(kind='scatter', x='X8', y='Y1', ax=axs[7])
    plt.savefig('heating_scatter_plot.png')
    #For cooling load
    fig, axs = plt.subplots(1, 8, sharey=True)
    normalized_data.plot(kind='scatter', x='X1', y='Y2', ax=axs[0], figsize=(16, 8))
    normalized_data.plot(kind='scatter', x='X2', y='Y2', ax=axs[1])
    normalized_data.plot(kind='scatter', x='X3', y='Y2', ax=axs[2])
    normalized_data.plot(kind='scatter', x='X4', y='Y2', ax=axs[3])
    normalized_data.plot(kind='scatter', x='X5', y='Y2', ax=axs[4])
    normalized_data.plot(kind='scatter', x='X6', y='Y2', ax=axs[5])
    normalized_data.plot(kind='scatter', x='X7', y='Y2', ax=axs[6])
    normalized_data.plot(kind='scatter', x='X8', y='Y2', ax=axs[7])
    plt.savefig('cooling_scatter_plot.png')
    plt.show()
    sns.heatmap(X.corr(), annot=True)
    plt.show()

# Working with models
# We start with a rough model which is a linear model of degree 1, i.e Linear Model


def display_scores(scores):
    print("\tScores:", scores)    #Scores for running the model on each different folds as validation sets
    print("\tMSE:", scores.mean())
    print("\tStandard deviation:", scores.std())

# Now we have our training set inputs and outputs, we can fit the training set
# with two linear regression models for our y1 and y2
def linear_regression():
    lin_reg = LinearRegression()
    lin_reg2 = LinearRegression()
    lin_reg.fit(X_train, y_train1)
    lin_reg2.fit(X_train, y_train2)

    # Computing the mean squared error on the training set w.r.t heating load
    energy_predictions = lin_reg.predict(X_train)
    lin_mse = mean_squared_error(y_train1, energy_predictions)
    print("------------------------------------------")
    print("Linear Regression")
    print("------------------------------------------\n")
    print("Heating Load")
    print("------------------------------------------")
    print("Training Set MSE:", lin_mse)

    # Now we try to find out how well this model generalize by testing
    # using Cross validation with 10 folds and computing the mean squared error on the validation set
    print("Validation Set Scores")
    lin1_scores = cross_val_score(lin_reg, X_train, y_train1, scoring="neg_mean_squared_error", cv=10)
    display_scores(-lin1_scores)

    # Computing the mean squared error on the training set w.r.t cooling load
    energy_predictions2 = lin_reg2.predict(X_train)
    lin_mse2 = mean_squared_error(y_train2, energy_predictions2)
    print("\nCooling Load")
    print("------------------------------------------")
    print("Training Set MSE:", lin_mse2)

    # Now we try to find out how well this model generalize by testing
    # using Cross validation with 10 folds and computing the mean squared error on the validation set
    print ("Cooling Load Validation Set Scores")
    lin2_scores = cross_val_score(lin_reg2, X_train, y_train2, scoring="neg_mean_squared_error", cv=10)
    display_scores(-lin2_scores)

# Function to plot learning curves for training and validation set
def plot_learning_curves(model, X, y):
    '''
    A function that plot the learning curve using training and validation errors

    model- the Machine Learning model
    X- input of the dataset (X-values)
    Y- output of the dataset (Y-values)
    '''

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for i in range(1, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        y_train_predict = model.predict(X_train[:i])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:i]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")

# Now that we have some good figures for the errors, we try to develop a model of degree 3 and
# and see if we can get a lower training and validation error without overfitting or underfitting the data
# we use both ridge regularization to compare and see which one best fits the training and validation set


def polynomial_regression():
    poly_features = PolynomialFeatures(degree=4, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)
    pol_reg1 = Ridge(alpha=0.1)
    pol_reg2 = Ridge(alpha=20.0)
    pol_reg1.fit(X_poly, y_train1)
    pol_reg2.fit(X_poly, y_train2)

    # Computing the mean squared error on the training and validation set w.r.t heating load
    energy_predictions3 = pol_reg1.predict(X_poly)
    pol_mse1 = mean_squared_error(y_train1, energy_predictions3)
    print("------------------------------------------")
    print("Polynomial Regression")
    print("------------------------------------------\n")
    print("Heating Load")
    print("------------------------------------------")
    print("Training Set MSE:", pol_mse1)

    pol1_scores = cross_val_score(pol_reg1, X_poly, y_train1, scoring="neg_mean_squared_error", cv=10)
    print ("Validation Set Scores:")
    display_scores(-pol1_scores)

    # Computing the mean squared error on the training and validation set w.r.t cooling load
    energy_predictions4 = pol_reg2.predict(X_poly)
    pol_mse2 = mean_squared_error(y_train2, energy_predictions4)
    print("\nCooling Load")
    print("------------------------------------------")
    print("Training Set MSE:", pol_mse2)

    pol2_scores = cross_val_score(pol_reg2, X_poly, y_train2, scoring="neg_mean_squared_error", cv=10)
    print ("Validation Set Scores:")
    display_scores(-pol2_scores)

    # For more evaluation using Learning Curves
    # plot_learning_curves(pol_reg1, X_train, y_train1)
    # plt.savefig('Learning curve1.png')
    # plt.show()
    # plot_learning_curves(pol_reg2, X_train, y_train2)
    # plt.savefig('Learning curve2.png')
    # plt.show()

    return poly_features,pol_reg1, pol_reg2


# Now, we can comfortably test our model on the test data with optimism we'll have good generalization
X_te = test_set.drop("cooling_load", axis=1)
X_test = X_te.drop("heating_load", axis=1)
y_test1 = test_set["heating_load"].copy()
y_test2 = test_set["cooling_load"].copy()

def testing():
    poly_features, pol_reg1, pol_reg2 =  polynomial_regression()
    X_poly_test = poly_features.fit_transform(X_test)
    test_predictions1 = pol_reg1.predict(X_poly_test)
    test_predictions2 = pol_reg2.predict(X_poly_test)
    test_mse1 = mean_squared_error(y_test1, test_predictions1)
    test_mse2 = mean_squared_error(y_test2, test_predictions2)
    print("------------------------------------------")
    print("Testing Data MSEs\n")
    print("Heating Load Test Set MSE:", test_mse1)
    print("Cooling Load Test Set MSE:", test_mse2)


def random_forest():
    rnd_reg1 = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    rnd_reg2 = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    rnd_reg1.fit(X_train, y_train1)
    rnd_reg2.fit(X_train, y_train2)
    y_pred_rf1 = rnd_reg1.predict(X_test)
    y_pred_rf2 = rnd_reg2.predict(X_test)
    rf_mse1 = mean_squared_error(y_test1, y_pred_rf1)
    rf_mse2 = mean_squared_error(y_test1, y_pred_rf2)
    print("\n------------------------------------------")
    print("Random Forest")
    print("------------------------------------------")
    print("Heating Load Training Set MSE:", rf_mse1)
    print("Cooling Load Training Set MSE:", rf_mse2)

# data_description()
# visualization()
# linear_regression()
# polynomial_regression()
# testing()
# random_forest()
