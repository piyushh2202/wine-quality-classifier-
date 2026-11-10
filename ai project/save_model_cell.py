# Add this as a NEW CELL at the END of your notebook and run it.
# Then download wine_model.pkl and scaler.pkl from the Kaggle/Colab file panel.
import joblib

joblib.dump(gs_cv.best_estimator_, "wine_model.pkl")   # best Logistic Regression
joblib.dump(scaler, "scaler.pkl")                      # the StandardScaler
print("Saved!")
