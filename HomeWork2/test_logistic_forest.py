from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def test(X, y):
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Логистическая регрессия
  log_reg = LogisticRegression()
  log_reg.fit(X_train_scaled, y_train)

  # Случайный лес
  rf = RandomForestClassifier()
  rf.fit(X_train_scaled, y_train)

  # Предсказания логистической регрессии
  y_pred_log_reg = log_reg.predict(X_test_scaled)
  print("Точность логистической регрессии (Logistic Regression): ", accuracy_score(y_test, y_pred_log_reg))

  # Предсказания случайного леса
  y_pred_rf = rf.predict(X_test_scaled)
  print("Точность случайного леса (Random Forest): ", accuracy_score(y_test, y_pred_rf))

  # Детальный отчёт по классификации
  print("\nОтчет по классификации случайного леса (Random Forest):\n", classification_report(y_test, y_pred_rf))