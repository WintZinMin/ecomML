import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest

LOG1P = np.log1p

def main():
    ecommerce = pd.read_csv('ecommerce_customer_behavior_dataset_v2.csv')

    ecommerce_df = remove_outliers(ecommerce)

    X = ecommerce_df[['Product_Category','Quantity','Payment_Method','City']]
    y = ecommerce_df['Total_Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    numerical_cols = X.select_dtypes(include='number').columns
    categorical_cols = X.select_dtypes(include='object').columns

    log_transformer = FunctionTransformer(LOG1P, validate=False)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    numerical_pipeline = Pipeline(steps=[
        ('log', log_transformer),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ('encoder', TargetEncoder()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge()),
    ])

    params = {
        'regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
    }

    grid = GridSearchCV(
        pipeline,
        params,
        cv=5,
        scoring='r2',
        n_jobs=-1,
    )

    grid.fit(X_train, y_train_log)
    model = grid.best_estimator_

    prediction = model.predict(X_test)
    train_prediction = model.predict(X_train)

    mse = mean_squared_error(y_test_log, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_log, prediction)

    train_mse = mean_squared_error(y_train_log, train_prediction)
    train_r2 = r2_score(y_train_log, train_prediction)
    train_rmse = np.sqrt(train_mse)

    print('Best Paramter:', grid.best_params_)
    print(f'Test MSE:{mse:.4f}')
    print(f'Train MSE: {train_mse:.4f}')
    print(f'Train RMSE: {train_rmse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test R2 : {r2:.4f}')
    print(f'Train R2 : {train_r2:.4f}')

    new_data = pd.DataFrame({
        'Quantity': [5, 8],
        'Product_Category': ['Books', 'Sports'],
        'Payment_Method': ['Credit Card', 'Digital Wallet'],
        'City': ['Istanbul', 'UK'],
    })

    log_pred = model.predict(new_data)
    original_pred = np.expm1(log_pred)

    for i, value in enumerate(original_pred, start=1):
        print(f'Predicted Amount for Person {i} : {value:.4f}')

    residuals = y_test_log - prediction
    stat, p = normaltest(residuals)
    if p > 0.05:
        print('Model is likely bias')
    else:
        print('Model is likely unbias')

    sns.histplot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.show()

    plt.scatter(prediction, residuals)
    plt.axhline(0, color='red')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predictions')
    plt.show()

    import scipy.stats as stats
    stats.probplot(residuals, dist='norm', plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    categorical_features = np.array([f"cat__{c}" for c in categorical_cols])
    numerical_features = np.array([f"num__{c}" for c in numerical_cols])
    all_features = np.concatenate([numerical_features, categorical_features])

    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.named_steps['regressor'].coef_,
    }).sort_values(by='importance', ascending=False)

    print(feature_importance.head())
    sns.barplot(data=feature_importance, y='feature', x='importance', hue='feature', palette='Set2')
    plt.title('Feature Importance')
    plt.show()

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'ridge_pipeline.joblib'))
    meta = {
        'best_params': grid.best_params_,
        'metrics': {
            'test_mse': float(mse),
            'test_rmse': float(rmse),
            'test_r2': float(r2),
            'train_mse': float(train_mse),
            'train_rmse': float(train_rmse),
            'train_r2': float(train_r2),
        },
        'features': feature_importance.to_dict(orient='records'),
        'input_schema': {
            'numeric': {
                'Quantity': {
                    'min': float(X['Quantity'].min()),
                    'max': float(X['Quantity'].max()),
                }
            },
            'categorical': {
                'Product_Category': sorted(X['Product_Category'].unique().tolist()),
                'Payment_Method': sorted(X['Payment_Method'].unique().tolist()),
                'City': sorted(X['City'].unique().tolist()),
            },
            'counts': {
                'Product_Category': int(X['Product_Category'].nunique()),
                'Payment_Method': int(X['Payment_Method'].nunique()),
                'City': int(X['City'].nunique()),
            }
        }
    }
    pd.Series(meta).to_json(os.path.join('models', 'ridge_metadata.json'))
    print('Saved model to models/ridge_pipeline.joblib and metadata to models/ridge_metadata.json')

def remove_outliers(ecommerce: pd.DataFrame) -> pd.DataFrame:
    df = ecommerce.copy()
    numerical_col = df.select_dtypes(include='number').columns
    for col in numerical_col:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

if __name__ == '__main__':
    main()