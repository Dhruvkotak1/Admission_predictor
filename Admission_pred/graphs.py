from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
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
model2 = GradientBoostingRegressor()
model3 = RandomForestRegressor()
model4 = KNeighborsRegressor()
model5 = DecisionTreeRegressor()

model.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)


y_pred = model.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)
y_pred4 = model4.predict(x_test)
y_pred5 = model5.predict(x_test)
Acuracy = r2_score(y_test,y_pred)
print("Accuracy by Linear Regression: ", Acuracy)
Acuracy2 = r2_score(y_test,y_pred2)
print("Accuracy by Gradient Boosting: ", Acuracy2)
Acuracy3 = r2_score(y_test,y_pred3)
print("Accuracy by Random Forest: ", Acuracy3)
Acuracy4 = r2_score(y_test,y_pred4)
print("Accuracy by K Neighbors: ", Acuracy4)
Acuracy5 = r2_score(y_test,y_pred5)
print("Accuracy by Decision Tree: ", Acuracy5)
# Plot the results
x_axis = ["LinearRegression", "GradientBoosting", "RandomForest", "KNN", "Decision Tree"]
y_axis = [Acuracy, Acuracy2, Acuracy3, Acuracy4, Acuracy5]
plt.bar(x_axis, y_axis,color=['blue', 'orange', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Performance of Different Models')
plt.xticks(rotation=45)

plt.ylim(0, 1)
plt.show()

# Save the model
#joblib.dump(model, 'admission_prediction_model')