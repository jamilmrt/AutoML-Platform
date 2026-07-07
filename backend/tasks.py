


import os
import pandas as pd
import numpy as np
import joblib
from celery import Celery
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.cluster import KMeans

#  Additional imports for profiling for analysis and insights
from ydata_profiling import ProfileReport

# Initialize Celery app (connects to the Redis broker from docker-compose)
celery_app = Celery('automl_tasks', broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))

@celery_app.task(bind=True)
def train_model_task(self, file_path, target_column, job_id):
    """
    Celery task that reads a dataset, cleans it, and trains the best model 
    (Clustering, Classification, or Regression).
    """
    try:
        self.update_state(state='TRAINING', meta={'message': 'Loading data'})
        
        # 1. Load Data
        df = pd.read_csv(file_path)
        
        # --- NEW: GENERATE PANDAS PROFILING REPORT ---
        self.update_state(state='TRAINING', meta={'message': 'Generating Data Profiling Report'})
        
        # Build the profile report (explorative=True enables deeper multi-variable insights)
        profile = ProfileReport(df, title="AutoML Data Profiling Report", explorative=True)
        
        # Save it directly to our shared docker volume
        report_filename = f"/app/shared_data/report_{job_id}.html"
        profile.to_file(report_filename)
        # ---------------------------------------------
        
        # Save JSON for PowerBI / External BI Tools
        json_filename = f"/app/shared_data/report_{job_id}.json"
        profile.to_file(json_filename)
        # ---------------------------------------------
        
        # 2. Automated Feature Engineering & Cleaning Setup
        # We define this first because all 3 paths (Clustering, Classification, Regression) need it.
        X_temp = df.drop(columns=[target_column]) if target_column else df.copy()
        
        numeric_features = X_temp.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_temp.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model_insights = {}

        # ==========================================
        # PATH A: UNSUPERVISED ROUTE (K-MEANS)
        # ==========================================
        if not target_column:
            self.update_state(state='TRAINING', meta={'message': 'Running Unsupervised Clustering'})
            X = df.copy()
            
            best_k = 2
            best_score = -1
            best_pipeline = None
            
            for k in range(2, 6):
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('clusterer', KMeans(n_clusters=k, random_state=42, n_init='auto'))
                ])
                
                X_transformed = pipeline.named_steps['preprocessor'].fit_transform(X)
                pipeline.fit(X)
                labels = pipeline.named_steps['clusterer'].labels_
                
                score = silhouette_score(X_transformed, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_pipeline = pipeline

            # Extract Cluster Sizes
            labels = best_pipeline.named_steps['clusterer'].labels_
            unique, counts = np.unique(labels, return_counts=True)
            model_insights['cluster_sizes'] = [{"name": f"Cluster {str(u)}", "size": int(c)} for u, c in zip(unique, counts)]

            model_filename = f"/app/shared_data/model_{job_id}.joblib"
            joblib.dump(best_pipeline, model_filename)
            
            return {
                'status': 'COMPLETED',
                'winning_model': f'K-Means (k={best_k})',
                'accuracy': round(best_score, 4),
                'model_path': model_filename,
                'insights': model_insights,
            'has_report': True # <-- Let the frontend know a report exists
            }

        # ==========================================
        # PATH B: SUPERVISED ROUTE (CLASS. vs REG.)
        # ==========================================
        else:
            self.update_state(state='TRAINING', meta={'message': 'Training supervised models'})
            
            df = df.dropna(subset=[target_column])
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Use Scikit-learn's native checker instead of guessing by unique values
            from sklearn.utils.multiclass import type_of_target
            is_regression = type_of_target(y) == 'continuous'
            
            if is_regression:
                models = {
                    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
                    'Random Forest Regressor': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
                }
            else:
                models = {
                    'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))]),
                    'Random Forest Classifier': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
                }
            
            best_model_name = ""
            best_score = -float('inf')
            best_pipeline = None
            
            for name, pipeline in models.items():
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                
                # Use R-squared for regression, Accuracy for classification
                if is_regression:
                    score = r2_score(y_test, predictions)
                else:
                    score = accuracy_score(y_test, predictions)
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_pipeline = pipeline

            # Extract Feature Importance (works for both RF Classifier and Regressor)
            if 'Random Forest' in best_model_name:
                step_name = 'regressor' if is_regression else 'classifier'
                feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
                importances = best_pipeline.named_steps[step_name].feature_importances_
                
                importance_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
                model_insights['feature_importance'] = [{"name": name, "value": float(score)} for name, score in importance_data]

            model_filename = f"/app/shared_data/model_{job_id}.joblib"
            joblib.dump(best_pipeline, model_filename)
            
            return {
                'status': 'COMPLETED',
                'winning_model': best_model_name,
                'accuracy': round(best_score, 4), 
                'model_path': model_filename,
                'insights': model_insights
            }
            
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}