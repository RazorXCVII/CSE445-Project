import pandas as pd
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data using pandas
file_path = os.path.join("data", "customer_survey.csv")  # Add the correct file path
df = pd.read_csv(file_path)

#Data Attributes
X = df[['Age','Gender','Income', 'AverageSpend','PreferredCuisine','TimeOfVisit','GroupSize','DiningOccasion','MealType','OnlineReservation','DeliveryOrder','LoyaltyProgramMember','WaitTime','ServiceRating','FoodRating','AmbianceRating','HighSatisfaction']]


#Target Attribute
y = df['WillVisit']

#Splitting Training and Testing data by 90%-10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.1, random_state=28)

#Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report For Gaussian Naive Bayes: ")
print(y_pred)
print("")
print(classification_report(y_test, y_pred))
print("")

#Gini Impurity
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred2 = dtc.predict(X_test)
print("Classification Report For CART (Gini Impurity): ")
print(y_pred2)
print("")
print(classification_report(y_test, y_pred2))
print("")

#Entropy
dtc2 = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.04)

dtc2 = DecisionTreeClassifier()
dtc2.fit(X_train, y_train)

y_pred3 = dtc2.predict(X_test)
print("Classification Report For ID3 (Entropy): ")
print(y_pred3)
print("")
print(classification_report(y_test, y_pred3))
print("")


print("Feature Importance Score: ")
features = pd.DataFrame(dtc.feature_importances_, index = X.columns)
print(features.head(27))