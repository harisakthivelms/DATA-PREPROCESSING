## DATA PREPROCESSING
# AIM

To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset into training and testing sets, and applying feature scaling.

# PROCEDURE
Import the required Python libraries.
Mount Google Drive and load the dataset using Pandas.
Display the first few records of the dataset.
Inspect the dataset using df.info() and df.shape.
Separate the independent variables (X) and dependent variable (Y).
Convert the independent variables into an array.
Identify and handle missing values using SimpleImputer with the mean strategy.
Encode the categorical Country column using LabelEncoder.
Apply One-Hot Encoding to convert categorical country values into dummy variables.
Encode the dependent variable Purchased using LabelEncoder.
Split the dataset into training and testing sets using train_test_split.
Apply StandardScaler for feature scaling.
Display the preprocessed training and testing datasets.
## PROGRAM
import numpy as np
import pandas as pd

# Import required libraries
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/Data.csv')

# Display first five records
print("First five records:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display shape
print("\nDataset Shape:")
print(df.shape)

# Separate independent and dependent variables
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

print("\nIndependent Variables (X):")
print(X)

print("\nDependent Variable (Y):")
print(Y)

# Handle missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("\nAfter Handling Missing Values:")
print(X)

# Encode categorical Country column using LabelEncoder
label_encoder = LabelEncoder()
X[:, 0] = label_encoder.fit_transform(X[:, 0])

print("\nAfter Label Encoding:")
print(X)

# One-Hot Encoding for Country column
one_hot_encoder = OneHotEncoder(
    categories='auto',
    sparse_output=False
)

country_encoded = one_hot_encoder.fit_transform(X[:, [0]])

# Combine encoded country with remaining columns
X = np.concatenate((country_encoded, X[:, 1:]), axis=1)

print("\nAfter One-Hot Encoding:")
print(X)

# Encode dependent variable Purchased
Y = label_encoder.fit_transform(Y)

print("\nEncoded Dependent Variable:")
print(Y)

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=0
)

# Apply Standard Scaling
scaler = StandardScaler()

X_train[:, 3:] = scaler.fit_transform(X_train[:, 3:])
X_test[:, 3:] = scaler.transform(X_test[:, 3:])

# Display preprocessed datasets
print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\nY_train:")
print(Y_train)

print("\nY_test:")
print(Y_test)
Note

If your Scikit-learn version gives an error for:

sparse_output=False

replace it with:

sparse=False

This preprocessing workflow is based on the common Country, Age, Salary, Purchased dataset used to demonstrate missing-value imputation, categorical encoding, train/test splitting, and feature scaling.

## RESULT

The dataset was successfully preprocessed using Python and Scikit-learn. Missing values were handled using mean imputation, categorical data was converted into numerical form using Label Encoding and One-Hot Encoding, and the dependent variable was encoded using LabelEncoder. The dataset was then divided into training and testing sets, and numerical features were standardized using StandardScaler.
