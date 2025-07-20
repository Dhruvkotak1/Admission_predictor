from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load the data and split it into train and test
data = pd.read_csv('C:\\Users\\HP\\Desktop\\admission_data.csv')
x = data.drop('Chance of Admit', axis=1)
y = data['Chance of Admit']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)
x_train = x_train.values
y_train = y_train.values
x_test = x_test.values    
y_test = y_test.values

# Train the model and measure accuracy
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
Accuracy = r2_score(y_test,y_pred)
print(Accuracy)

# Save the model
joblib.dump(model, 'admission_prediction_model')