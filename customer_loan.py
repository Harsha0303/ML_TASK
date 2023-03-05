import pandas as pd

df = pd.read_csv("customer_loan_data.csv")
from sklearn.model_selection import train_test_split

# Separate the target variable from the features
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define the column transformer for scaling numerical features and encoding categorical features
num_features = ["loan_amount", "annual_income"]
cat_features = ["credit_score", "employment_type"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(), cat_features)
])

# Preprocess the training data
X_train = preprocessor.fit_transform(X_train)

# Preprocess the testing data
X_test = preprocessor.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
