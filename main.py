import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("telecom.csv", sep=';')

# Rename columns
df.columns = ['State','Account Code','Area Code','Phone','International Plan','VMail Plan',
              'VMail Message','Day Mins','Day Calls','Day Charge','Eve Mins','Eve Calls',
              'Eve Charge','Night Mins','Night Calls','Night Charge','International Mins',
              'International calls','International charge','CustServ Calls','Churn']

# Encode categorical data
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Split data
X = df.drop('Churn', axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Risk score
df['CHURN_RISK_SCORE'] = model.predict_proba(scaler.transform(X))[:, 1]
df['CHURN_FLAG'] = df['CHURN_RISK_SCORE'].apply(lambda x: 'YES' if x > 0.5 else 'NO')

print(df[['Churn','CHURN_RISK_SCORE','CHURN_FLAG']].head())
