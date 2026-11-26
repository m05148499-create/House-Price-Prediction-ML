import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train():
    print("Loading dataset...")
    # Make sure you have train.csv from Kaggle in this folder
    df = pd.read_csv('train.csv')

    # Map the Lovable screenshot features to Kaggle Ames features
    # OverallQual, GrLivArea, TotalBsmtSF, GarageCars, FullBath, YearBuilt, TotRmsAbvGrd, LotArea, Fireplaces, YearRemodAdd
    features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 
                'YearBuilt', 'TotRmsAbvGrd', 'LotArea', 'Fireplaces', 'YearRemodAdd']
    target = 'SalePrice'

    X = df[features].copy()
    y = df[target].copy()

    # Fill missing values
    X = X.fillna(X.median())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"R² Score: {r2:.3f}")
    print(f"MAE: ${mae:.2f}")

    # Save
    joblib.dump({'model': model, 'features': features}, 'house_price_model.pkl')
    print("Model saved!")

if __name__ == "__main__":
    train()
