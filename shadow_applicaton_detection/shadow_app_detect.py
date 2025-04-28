import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


data = pd.read_csv('network_logs_with_risk.csv')

features = ['is_new_domain', 'popularity_score', 'access_frequency', 'avg_upload_size', 'domain_risk_score']
target = 'label'

X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


new_data = pd.DataFrame({
    'is_new_domain': [1],
    'popularity_score': [5],
    'access_frequency': [2],
    'avg_upload_size': [1024],
    'domain_risk_score': [85]   # Very risky!
})
prediction = clf.predict(new_data)
print("Prediction (0 = safe, 1 = shadow app):", prediction[0])

