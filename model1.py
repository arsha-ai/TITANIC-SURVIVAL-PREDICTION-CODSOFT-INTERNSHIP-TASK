import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv(r"Titanic-Dataset.csv")

# Preprocess the data
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Split features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Define numeric and categorical columns
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

# Preprocess the data
X, y, preprocessor = preprocess_data(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessor and model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(random_state=42))])

# Fit the model
model.fit(X_train, y_train)

# Function to get user input
def get_user_input():
    print("\nEnter passenger information:")
    pclass = int(input("Passenger Class (1, 2, or 3): "))
    sex = input("Sex (male or female): ")
    age = float(input("Age: "))
    sibsp = int(input("Number of Siblings/Spouses Aboard: "))
    parch = int(input("Number of Parents/Children Aboard: "))
    fare = float(input("Fare: "))
    embarked = input("Port of Embarkation (S, C, or Q): ")
    
    return pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], 
                        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Function to make prediction
def predict_survival(passenger_data):
    prediction = model.predict(passenger_data)
    probability = model.predict_proba(passenger_data)[0][1]
    return prediction[0], probability

# Main loop
while True:
    user_input = get_user_input()
    prediction, probability = predict_survival(user_input)
    
    print("\nSurvival Prediction:")
    if prediction == 1:
        print(f"The passenger would likely SURVIVE with a probability of {probability:.2f}")
    else:
        print(f"The passenger would likely NOT SURVIVE with a probability of {1-probability:.2f}")
    
    again = input("\nWould you like to make another prediction? (yes/no): ")
    if again.lower() != 'yes':
        break

print("Thank you for using the Titanic Survival Prediction model!")
