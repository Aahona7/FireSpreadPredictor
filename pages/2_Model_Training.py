import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import joblib
import os
from utils.models import train_regression_models, train_classification_models
from utils.preprocessing import preprocess_data, create_features

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Model Training & Evaluation")

# Load data
if 'fire_data' not in st.session_state:
    st.error("Please load data from the main page first.")
    st.stop()

data = st.session_state.fire_data

if data is None:
    st.error("No data available.")
    st.stop()

# Model configuration sidebar
st.sidebar.header("Model Configuration")

# Problem type selection
problem_type = st.sidebar.radio(
    "Select Problem Type",
    ["Regression (Predict Burned Area)", "Classification (Fire Risk Level)"],
    help="Choose whether to predict continuous burned area or classify fire risk"
)

# Feature selection
st.sidebar.subheader("Feature Selection")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Remove target variables from feature options
exclude_cols = ['area'] if problem_type.startswith("Regression") else []
available_features = [col for col in numeric_cols if col not in exclude_cols]

selected_features = st.sidebar.multiselect(
    "Select Features",
    available_features,
    default=available_features[:8] if len(available_features) > 8 else available_features,
    help="Choose input features for the model"
)

# Add categorical features if available
if categorical_cols:
    categorical_features = st.sidebar.multiselect(
        "Select Categorical Features",
        categorical_cols,
        help="Categorical features will be encoded automatically"
    )
else:
    categorical_features = []

# Advanced options
with st.sidebar.expander("Advanced Options"):
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 9999, 42)
    scale_features = st.checkbox("Scale Features", value=True)

# Main content
if len(selected_features) == 0:
    st.warning("Please select at least one feature to train models.")
    st.stop()

# Prepare data
with st.spinner("Preparing data for training..."):
    try:
        # Create enhanced features
        enhanced_data = create_features(data.copy())
        
        # Select features
        all_features = selected_features + categorical_features
        available_features_in_data = [f for f in all_features if f in enhanced_data.columns]
        
        if len(available_features_in_data) == 0:
            st.error("No selected features found in the dataset.")
            st.stop()
        
        X = enhanced_data[available_features_in_data].copy()
        
        # Handle categorical variables
        categorical_encoders = {}
        for cat_col in categorical_features:
            if cat_col in X.columns:
                le = LabelEncoder()
                X[cat_col] = le.fit_transform(X[cat_col].astype(str))
                categorical_encoders[cat_col] = le
        
        # Prepare target variable
        if problem_type.startswith("Regression"):
            y = enhanced_data['area'].copy()
            # Log transform for area (common for fire data)
            y_transformed = np.log1p(y)  # log(1 + area) to handle zeros
            target_name = "Log(Burned Area + 1)"
        else:
            # Create fire risk categories
            def categorize_risk(area):
                if area == 0:
                    return 0  # No fire
                elif area <= 1:
                    return 1  # Low risk
                elif area <= 10:
                    return 2  # Medium risk
                else:
                    return 3  # High risk
            
            y = enhanced_data['area'].apply(categorize_risk)
            y_transformed = y
            target_name = "Fire Risk Category"
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | pd.isnull(y_transformed))
        X = X[mask]
        y_transformed = y_transformed[mask]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=test_size, random_state=random_state, stratify=y_transformed if problem_type.startswith("Classification") else None
        )
        
        # Feature scaling
        scaler = StandardScaler() if scale_features else None
        if scale_features:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        st.success(f"✅ Data prepared: {len(X)} samples, {len(selected_features)} features")
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        st.stop()

# Display data preparation summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", len(X))
with col2:
    st.metric("Training Samples", len(X_train))
with col3:
    st.metric("Test Samples", len(X_test))

# Model training section
st.header("Model Training")

if st.button("Train Models", type="primary"):
    with st.spinner("Training models... This may take a few minutes."):
        try:
            if problem_type.startswith("Regression"):
                models, metrics = train_regression_models(
                    X_train_scaled, X_test_scaled, y_train, y_test
                )
                
                # Store models and preprocessing objects
                st.session_state.trained_models = models
                st.session_state.model_metrics = metrics
                st.session_state.scaler = scaler
                st.session_state.feature_names = available_features_in_data
                st.session_state.categorical_encoders = categorical_encoders
                st.session_state.problem_type = "regression"
                
                st.success("✅ Regression models trained successfully!")
                
            else:
                models, metrics = train_classification_models(
                    X_train_scaled, X_test_scaled, y_train, y_test
                )
                
                # Store models and preprocessing objects
                st.session_state.trained_models = models
                st.session_state.model_metrics = metrics
                st.session_state.scaler = scaler
                st.session_state.feature_names = available_features_in_data
                st.session_state.categorical_encoders = categorical_encoders
                st.session_state.problem_type = "classification"
                
                st.success("✅ Classification models trained successfully!")
                
        except Exception as e:
            st.error(f"Error training models: {str(e)}")

# Display results if models are trained
if 'trained_models' in st.session_state and 'model_metrics' in st.session_state:
    
    st.header("Model Performance")
    
    models = st.session_state.trained_models
    metrics = st.session_state.model_metrics
    
    # Performance comparison table
    metrics_df = pd.DataFrame(metrics).T
    st.subheader("Performance Comparison")
    st.dataframe(metrics_df.round(4), use_container_width=True)
    
    # Best model highlight
    if st.session_state.problem_type == "regression":
        best_model = metrics_df['R² Score'].idxmax()
        best_score = metrics_df.loc[best_model, 'R² Score']
        st.success(f"🏆 Best performing model: **{best_model}** (R² Score: {best_score:.4f})")
    else:
        best_model = metrics_df['Accuracy'].idxmax()
        best_score = metrics_df.loc[best_model, 'Accuracy']
        st.success(f"🏆 Best performing model: **{best_model}** (Accuracy: {best_score:.4f})")
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🎯 Predictions vs Actual", "🔍 Feature Importance"])
    
    with tab1:
        if st.session_state.problem_type == "regression":
            # Regression metrics visualization
            fig_metrics = make_subplots(
                rows=1, cols=2,
                subplot_titles=('R² Score Comparison', 'RMSE Comparison')
            )
            
            models_list = list(metrics.keys())
            r2_scores = [metrics[model]['R² Score'] for model in models_list]
            rmse_scores = [metrics[model]['RMSE'] for model in models_list]
            
            fig_metrics.add_trace(
                go.Bar(x=models_list, y=r2_scores, name='R² Score', marker_color='blue'),
                row=1, col=1
            )
            
            fig_metrics.add_trace(
                go.Bar(x=models_list, y=rmse_scores, name='RMSE', marker_color='red'),
                row=1, col=2
            )
            
            fig_metrics.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
        else:
            # Classification metrics visualization
            models_list = list(metrics.keys())
            accuracy_scores = [metrics[model]['Accuracy'] for model in models_list]
            precision_scores = [metrics[model]['Precision'] for model in models_list]
            recall_scores = [metrics[model]['Recall'] for model in models_list]
            f1_scores = [metrics[model]['F1 Score'] for model in models_list]
            
            fig_class_metrics = go.Figure()
            fig_class_metrics.add_trace(go.Bar(name='Accuracy', x=models_list, y=accuracy_scores))
            fig_class_metrics.add_trace(go.Bar(name='Precision', x=models_list, y=precision_scores))
            fig_class_metrics.add_trace(go.Bar(name='Recall', x=models_list, y=recall_scores))
            fig_class_metrics.add_trace(go.Bar(name='F1 Score', x=models_list, y=f1_scores))
            
            fig_class_metrics.update_layout(
                title='Classification Metrics Comparison',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_class_metrics, use_container_width=True)
    
    with tab2:
        # Predictions vs actual for selected model
        selected_model_name = st.selectbox("Select Model for Prediction Analysis", list(models.keys()))
        selected_model = models[selected_model_name]
        
        # Make predictions
        if st.session_state.problem_type == "regression":
            y_pred = selected_model.predict(X_test_scaled)
            
            # Scatter plot of predictions vs actual
            fig_pred = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': f'Actual {target_name}', 'y': f'Predicted {target_name}'},
                title=f'{selected_model_name}: Predictions vs Actual'
            )
            
            # Add perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Residuals plot
            residuals = y_test - y_pred
            fig_residuals = px.scatter(
                x=y_pred, y=residuals,
                labels={'x': f'Predicted {target_name}', 'y': 'Residuals'},
                title=f'{selected_model_name}: Residuals Plot'
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
            
        else:
            y_pred = selected_model.predict(X_test_scaled)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f'{selected_model_name}: Confusion Matrix',
                labels={'x': 'Predicted', 'y': 'Actual'}
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab3:
        # Feature importance (for tree-based models)
        selected_model_name = st.selectbox(
            "Select Model for Feature Importance", 
            list(models.keys()),
            key="importance_model"
        )
        selected_model = models[selected_model_name]
        
        if hasattr(selected_model, 'feature_importances_'):
            importance_scores = selected_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': available_features_in_data,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'{selected_model_name}: Feature Importance'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
        elif hasattr(selected_model, 'coef_'):
            # For linear models, show coefficients
            if st.session_state.problem_type == "regression":
                coef_scores = selected_model.coef_
            else:
                coef_scores = np.abs(selected_model.coef_[0] if len(selected_model.coef_.shape) > 1 else selected_model.coef_)
            
            coef_df = pd.DataFrame({
                'Feature': available_features_in_data,
                'Coefficient': coef_scores
            }).sort_values('Coefficient', ascending=True)
            
            fig_coef = px.bar(
                coef_df,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title=f'{selected_model_name}: Feature Coefficients'
            )
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info(f"{selected_model_name} does not support feature importance visualization.")

# Model export section
if 'trained_models' in st.session_state:
    st.header("Model Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Best Model"):
            try:
                # Create models directory if it doesn't exist
                os.makedirs("models", exist_ok=True)
                
                # Save the best model and preprocessing objects
                model_data = {
                    'models': st.session_state.trained_models,
                    'scaler': st.session_state.scaler,
                    'feature_names': st.session_state.feature_names,
                    'categorical_encoders': st.session_state.categorical_encoders,
                    'problem_type': st.session_state.problem_type,
                    'metrics': st.session_state.model_metrics
                }
                
                joblib.dump(model_data, "models/forest_fire_models.pkl")
                st.success("✅ Models saved successfully!")
                
            except Exception as e:
                st.error(f"Error saving models: {str(e)}")
    
    with col2:
        st.info("Models and preprocessing objects will be saved for use in prediction and simulation pages.")

# Cross-validation section
if 'trained_models' in st.session_state:
    with st.expander("Cross-Validation Analysis"):
        if st.button("Run Cross-Validation"):
            from sklearn.model_selection import cross_val_score
            
            with st.spinner("Running cross-validation..."):
                cv_results = {}
                
                for model_name, model in models.items():
                    if st.session_state.problem_type == "regression":
                        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    else:
                        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    
                    cv_results[model_name] = {
                        'Mean': scores.mean(),
                        'Std': scores.std(),
                        'Min': scores.min(),
                        'Max': scores.max()
                    }
                
                cv_df = pd.DataFrame(cv_results).T
                st.dataframe(cv_df.round(4), use_container_width=True)
                
                # Visualization
                fig_cv = px.bar(
                    x=list(cv_results.keys()),
                    y=[cv_results[model]['Mean'] for model in cv_results.keys()],
                    error_y=[cv_results[model]['Std'] for model in cv_results.keys()],
                    title='Cross-Validation Results (Mean ± Std)'
                )
                st.plotly_chart(fig_cv, use_container_width=True)
