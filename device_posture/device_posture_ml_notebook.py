import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.DataFrame([
    {"av": 1, "mdm": 1, "checkin_age": 10, "incidents": 0, "usb_insertions": 0,
     "clipboard_to_terminal": 0, "risky_apps_detected": 0,
     "os": 1, "os_version_age": 10, "disk_encrypted": 1, "admin_user": 0, "label": 0},
     
    {"av": 1, "mdm": 1, "checkin_age": 20, "incidents": 1, "usb_insertions": 2,
     "clipboard_to_terminal": 1, "risky_apps_detected": 0,
     "os": 1, "os_version_age": 20, "disk_encrypted": 1, "admin_user": 0, "label": 0},
     
    {"av": 0, "mdm": 0, "checkin_age": 300, "incidents": 4, "usb_insertions": 7,
     "clipboard_to_terminal": 5, "risky_apps_detected": 1,
     "os": 0, "os_version_age": 700, "disk_encrypted": 0, "admin_user": 1, "label": 1},
     
    {"av": 1, "mdm": 1, "checkin_age": 15, "incidents": 0, "usb_insertions": 0,
     "clipboard_to_terminal": 0, "risky_apps_detected": 0,
     "os": 1, "os_version_age": 5, "disk_encrypted": 1, "admin_user": 0, "label": 0},
     
    {"av": 0, "mdm": 0, "checkin_age": 250, "incidents": 3, "usb_insertions": 5,
     "clipboard_to_terminal": 4, "risky_apps_detected": 1,
     "os": 0, "os_version_age": 400, "disk_encrypted": 0, "admin_user": 1, "label": 1},
     
    {"av": 1, "mdm": 0, "checkin_age": 180, "incidents": 2, "usb_insertions": 4,
     "clipboard_to_terminal": 2, "risky_apps_detected": 0,
     "os": 1, "os_version_age": 100, "disk_encrypted": 0, "admin_user": 1, "label": 1}
])



X = data.drop(columns=["label"])
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

feature_importances = dict(zip(X.columns, clf.feature_importances_))
print("Feature Importances:\n")
for feature, importance in feature_importances.items():
    print(f"{feature}: {importance:.2f}")


sample_device = pd.DataFrame([{
    "av": 1,
    "mdm": 0,
    "checkin_age": 180,
    "incidents": 2,
    "usb_insertions": 4,
    "clipboard_to_terminal": 2,
    "risky_apps_detected": 0,
    "os": 1,
    "os_version_age": 100,
    "disk_encrypted": 0,
    "admin_user": 1
}])

risk_prediction = clf.predict(sample_device)[0]

print(f"Risk Prediction for Sample Device: {'Risky' if risk_prediction == 1 else 'Healthy'}")
