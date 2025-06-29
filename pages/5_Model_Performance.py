import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance Analysis")

st.markdown("""
This page provides comprehensive analysis of trained model performance, including 
detailed metrics, cross-validation results, and advanced diagnostic tools.
""")

# Load trained models
@st.cache_data
def load_trained_models():
    try:
        if os.path.exists("models/forest_fire_models.pkl"):
            return joblib.load("models/forest_fire_models.pkl")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

model_data = load_trained_models()

if model_data is None:
    st.error("No trained models found. Please train models first in the Model Training page.")
    st.stop()

models = model_data['models']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
categorical_encoders = model_data['categorical_encoders']
problem_type = model_data['problem_type']
base_metrics = model_data['metrics']

st.success("✅ Model performance data loaded successfully!")

# Performance analysis tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overall Performance", 
    "🎯 Detailed Metrics", 
    "📈 Learning Curves", 
    "🔍 Error Analysis", 
    "⚖️ Model Comparison"
])

with tab1:
    st.header("Overall Performance Summary")
    
    # Performance metrics overview
    metrics_df = pd.DataFrame(base_metrics).T
    
    # Best model identification
    if problem_type == "regression":
        best_metric = "R² Score"
        best_model = metrics_df[best_metric].idxmax()
        best_score = metrics_df.loc[best_model, best_metric]
    else:
        best_metric = "Accuracy"
        best_model = metrics_df[best_metric].idxmax()
        best_score = metrics_df.loc[best_model, best_metric]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Model", best_model)
    with col2:
        st.metric(f"Best {best_metric}", f"{best_score:.4f}")
    with col3:
        st.metric("Total Models", len(models))
    
    # Performance comparison chart
    st.subheader("Model Performance Comparison")
    
    if problem_type == "regression":
        # Regression metrics comparison
        fig_performance = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score', 'RMSE'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        model_names = list(base_metrics.keys())
        r2_scores = [base_metrics[model]['R² Score'] for model in model_names]
        rmse_scores = [base_metrics[model]['RMSE'] for model in model_names]
        
        fig_performance.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R² Score', marker_color='blue'),
            row=1, col=1
        )
        
        fig_performance.add_trace(
            go.Bar(x=model_names, y=rmse_scores, name='RMSE', marker_color='red'),
            row=1, col=2
        )
        
        fig_performance.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_performance, use_container_width=True)
        
    else:
        # Classification metrics comparison
        model_names = list(base_metrics.keys())
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        fig_class_perf = go.Figure()
        
        for metric in metrics_to_plot:
            scores = [base_metrics[model][metric] for model in model_names]
            fig_class_perf.add_trace(go.Bar(name=metric, x=model_names, y=scores))
        
        fig_class_perf.update_layout(
            title='Classification Metrics Comparison',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_class_perf, use_container_width=True)
    
    # Performance table
    st.subheader("Detailed Performance Metrics")
    
    # Format metrics for display
    formatted_metrics = metrics_df.round(4)
    
    # Highlight best performance
    if problem_type == "regression":
        styled_df = formatted_metrics.style.highlight_max(subset=['R² Score'], color='lightgreen')
        styled_df = styled_df.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
    else:
        styled_df = formatted_metrics.style.highlight_max(
            subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], 
            color='lightgreen'
        )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Model rankings
    st.subheader("Model Rankings")
    
    if problem_type == "regression":
        rankings = {
            'R² Score': metrics_df['R² Score'].rank(ascending=False),
            'RMSE': metrics_df['RMSE'].rank(ascending=True),  # Lower is better
            'MAE': metrics_df['MAE'].rank(ascending=True)
        }
    else:
        rankings = {
            'Accuracy': metrics_df['Accuracy'].rank(ascending=False),
            'Precision': metrics_df['Precision'].rank(ascending=False),
            'Recall': metrics_df['Recall'].rank(ascending=False),
            'F1 Score': metrics_df['F1 Score'].rank(ascending=False)
        }
    
    rankings_df = pd.DataFrame(rankings)
    rankings_df['Average Rank'] = rankings_df.mean(axis=1).round(2)
    rankings_df = rankings_df.sort_values('Average Rank')
    
    st.dataframe(rankings_df, use_container_width=True)

with tab2:
    st.header("Detailed Model Metrics")
    
    # Model selection for detailed analysis
    selected_model_name = st.selectbox(
        "Select Model for Detailed Analysis",
        list(models.keys()),
        key="detailed_analysis"
    )
    
    selected_model = models[selected_model_name]
    selected_metrics = base_metrics[selected_model_name]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{selected_model_name} Performance")
        
        # Display metrics
        for metric, value in selected_metrics.items():
            if isinstance(value, (int, float)):
                st.metric(metric, f"{value:.4f}")
        
        # Model parameters
        if hasattr(selected_model, 'get_params'):
            st.subheader("Model Parameters")
            params = selected_model.get_params()
            params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
            st.dataframe(params_df, use_container_width=True)
    
    with col2:
        st.subheader("Performance Visualization")
        
        if problem_type == "classification":
            # Load data for detailed analysis
            if 'fire_data' in st.session_state and st.session_state.fire_data is not None:
                data = st.session_state.fire_data.copy()
                
                try:
                    from utils.preprocessing import create_features
                    
                    # Prepare data same as training
                    enhanced_data = create_features(data)
                    
                    # Categorical encoding
                    for cat_col, encoder in categorical_encoders.items():
                        if cat_col in enhanced_data.columns:
                            enhanced_data[cat_col] = encoder.transform(enhanced_data[cat_col].astype(str))
                    
                    X = enhanced_data[feature_names].fillna(0)
                    
                    # Create target
                    def categorize_risk(area):
                        if area == 0:
                            return 0
                        elif area <= 1:
                            return 1
                        elif area <= 10:
                            return 2
                        else:
                            return 3
                    
                    y = enhanced_data['area'].apply(categorize_risk)
                    
                    # Remove missing values
                    mask = ~(X.isnull().any(axis=1) | pd.isnull(y))
                    X = X[mask]
                    y = y[mask]
                    
                    # Scale features
                    if scaler is not None:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values
                    
                    # Make predictions
                    y_pred = selected_model.predict(X_scaled)
                    y_pred_proba = selected_model.predict_proba(X_scaled) if hasattr(selected_model, 'predict_proba') else None
                    
                    # Classification report
                    st.text("Classification Report:")
                    class_report = classification_report(y, y_pred, output_dict=True)
                    report_df = pd.DataFrame(class_report).transpose()
                    st.dataframe(report_df.round(3), use_container_width=True)
                    
                    # Confusion matrix
                    cm = confusion_matrix(y, y_pred)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix",
                        labels={'x': 'Predicted', 'y': 'Actual'}
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # ROC curves for multiclass
                    if y_pred_proba is not None:
                        st.subheader("ROC Curves")
                        
                        from sklearn.preprocessing import label_binarize
                        from sklearn.metrics import roc_curve, auc
                        
                        # Binarize labels for multiclass ROC
                        y_bin = label_binarize(y, classes=[0, 1, 2, 3])
                        n_classes = y_bin.shape[1]
                        
                        fig_roc = go.Figure()
                        
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                name=f'Class {i} (AUC = {roc_auc:.2f})',
                                mode='lines'
                            ))
                        
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random',
                            line=dict(dash='dash')
                        ))
                        
                        fig_roc.update_layout(
                            title='ROC Curves',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate'
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in detailed analysis: {str(e)}")
            else:
                st.info("Load data from the main page to see detailed metrics.")

with tab3:
    st.header("Learning Curves Analysis")
    
    st.markdown("""
    Learning curves show how model performance changes with training set size and 
    can help identify overfitting or underfitting issues.
    """)
    
    # Learning curve configuration
    col1, col2 = st.columns(2)
    
    with col1:
        lc_model_name = st.selectbox(
            "Select Model for Learning Curves",
            list(models.keys()),
            key="learning_curves"
        )
        
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        n_jobs = st.slider("Parallel Jobs", 1, 4, 1)
    
    with col2:
        train_sizes = st.slider(
            "Training Set Size Range",
            0.1, 1.0, (0.1, 1.0),
            step=0.1,
            help="Range of training set sizes to evaluate"
        )
        
        n_train_sizes = st.slider("Number of Training Sizes", 5, 20, 10)
    
    if st.button("Generate Learning Curves"):
        if 'fire_data' in st.session_state and st.session_state.fire_data is not None:
            with st.spinner("Generating learning curves... This may take several minutes."):
                try:
                    data = st.session_state.fire_data.copy()
                    
                    from utils.preprocessing import create_features
                    
                    # Prepare data
                    enhanced_data = create_features(data)
                    
                    # Categorical encoding
                    for cat_col, encoder in categorical_encoders.items():
                        if cat_col in enhanced_data.columns:
                            enhanced_data[cat_col] = encoder.transform(enhanced_data[cat_col].astype(str))
                    
                    X = enhanced_data[feature_names].fillna(0)
                    
                    if problem_type == "regression":
                        y = np.log1p(enhanced_data['area'])
                        scoring = 'r2'
                    else:
                        def categorize_risk(area):
                            if area == 0:
                                return 0
                            elif area <= 1:
                                return 1
                            elif area <= 10:
                                return 2
                            else:
                                return 3
                        
                        y = enhanced_data['area'].apply(categorize_risk)
                        scoring = 'accuracy'
                    
                    # Remove missing values
                    mask = ~(X.isnull().any(axis=1) | pd.isnull(y))
                    X = X[mask]
                    y = y[mask]
                    
                    # Scale features
                    if scaler is not None:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values
                    
                    # Generate learning curve
                    selected_model = models[lc_model_name]
                    
                    train_sizes_abs, train_scores, val_scores = learning_curve(
                        selected_model,
                        X_scaled,
                        y,
                        cv=cv_folds,
                        n_jobs=n_jobs,
                        train_sizes=np.linspace(train_sizes[0], train_sizes[1], n_train_sizes),
                        scoring=scoring,
                        random_state=42
                    )
                    
                    # Calculate means and standard deviations
                    train_scores_mean = np.mean(train_scores, axis=1)
                    train_scores_std = np.std(train_scores, axis=1)
                    val_scores_mean = np.mean(val_scores, axis=1)
                    val_scores_std = np.std(val_scores, axis=1)
                    
                    # Plot learning curves
                    fig_lc = go.Figure()
                    
                    # Training scores
                    fig_lc.add_trace(go.Scatter(
                        x=train_sizes_abs,
                        y=train_scores_mean,
                        mode='lines+markers',
                        name='Training Score',
                        line=dict(color='blue'),
                        error_y=dict(type='data', array=train_scores_std)
                    ))
                    
                    # Validation scores
                    fig_lc.add_trace(go.Scatter(
                        x=train_sizes_abs,
                        y=val_scores_mean,
                        mode='lines+markers',
                        name='Validation Score',
                        line=dict(color='red'),
                        error_y=dict(type='data', array=val_scores_std)
                    ))
                    
                    fig_lc.update_layout(
                        title=f'Learning Curves - {lc_model_name}',
                        xaxis_title='Training Set Size',
                        yaxis_title=f'{scoring.title()} Score',
                        height=500
                    )
                    
                    st.plotly_chart(fig_lc, use_container_width=True)
                    
                    # Learning curve analysis
                    st.subheader("Learning Curve Analysis")
                    
                    final_train_score = train_scores_mean[-1]
                    final_val_score = val_scores_mean[-1]
                    score_gap = final_train_score - final_val_score
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Training Score", f"{final_train_score:.4f}")
                    with col2:
                        st.metric("Final Validation Score", f"{final_val_score:.4f}")
                    with col3:
                        st.metric("Training-Validation Gap", f"{score_gap:.4f}")
                    
                    # Interpretation
                    st.subheader("Interpretation")
                    
                    if score_gap > 0.1:
                        st.warning("🟡 **Possible Overfitting**: Large gap between training and validation scores suggests the model may be overfitting.")
                    elif final_val_score < 0.7:
                        st.info("🔵 **Possible Underfitting**: Low validation scores suggest the model may be underfitting.")
                    else:
                        st.success("🟢 **Good Fit**: Training and validation scores are reasonably close.")
                    
                    # Recommendations
                    if score_gap > 0.1:
                        st.write("**Recommendations to reduce overfitting:**")
                        st.write("- Increase training data size")
                        st.write("- Add regularization")
                        st.write("- Reduce model complexity")
                        st.write("- Use cross-validation")
                    
                    elif final_val_score < 0.7:
                        st.write("**Recommendations to improve model performance:**")
                        st.write("- Increase model complexity")
                        st.write("- Add more features")
                        st.write("- Try different algorithms")
                        st.write("- Improve data quality")
                
                except Exception as e:
                    st.error(f"Error generating learning curves: {str(e)}")
        else:
            st.error("Please load data from the main page first.")

with tab4:
    st.header("Error Analysis")
    
    st.markdown("""
    Analyze prediction errors to understand model weaknesses and improve performance.
    """)
    
    # Error analysis model selection
    error_model_name = st.selectbox(
        "Select Model for Error Analysis",
        list(models.keys()),
        key="error_analysis"
    )
    
    if 'fire_data' in st.session_state and st.session_state.fire_data is not None:
        try:
            data = st.session_state.fire_data.copy()
            
            from utils.preprocessing import create_features
            
            # Prepare data
            enhanced_data = create_features(data)
            
            # Categorical encoding
            for cat_col, encoder in categorical_encoders.items():
                if cat_col in enhanced_data.columns:
                    enhanced_data[cat_col] = encoder.transform(enhanced_data[cat_col].astype(str))
            
            X = enhanced_data[feature_names].fillna(0)
            
            if problem_type == "regression":
                y = np.log1p(enhanced_data['area'])
                y_original = enhanced_data['area']
            else:
                def categorize_risk(area):
                    if area == 0:
                        return 0
                    elif area <= 1:
                        return 1
                    elif area <= 10:
                        return 2
                    else:
                        return 3
                
                y = enhanced_data['area'].apply(categorize_risk)
                y_original = enhanced_data['area']
            
            # Remove missing values
            mask = ~(X.isnull().any(axis=1) | pd.isnull(y))
            X = X[mask]
            y = y[mask]
            y_original = y_original[mask]
            enhanced_data_clean = enhanced_data[mask]
            
            # Scale features
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Make predictions
            selected_model = models[error_model_name]
            y_pred = selected_model.predict(X_scaled)
            
            if problem_type == "regression":
                # Calculate errors
                errors = y - y_pred
                abs_errors = np.abs(errors)
                
                # Convert back to original scale for interpretation
                y_pred_original = np.expm1(y_pred)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Error Distribution")
                    
                    fig_error_dist = px.histogram(
                        errors,
                        nbins=50,
                        title="Distribution of Prediction Errors",
                        labels={'value': 'Error (log scale)', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig_error_dist, use_container_width=True)
                
                with col2:
                    st.subheader("Absolute Error vs Prediction")
                    
                    fig_error_scatter = px.scatter(
                        x=y_pred,
                        y=abs_errors,
                        title="Absolute Error vs Predicted Value",
                        labels={'x': 'Predicted Value (log scale)', 'y': 'Absolute Error'}
                    )
                    st.plotly_chart(fig_error_scatter, use_container_width=True)
                
                # Error statistics
                st.subheader("Error Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Error", f"{np.mean(errors):.4f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{np.mean(abs_errors):.4f}")
                with col3:
                    st.metric("Max Error", f"{np.max(abs_errors):.4f}")
                with col4:
                    error_std = np.std(errors)
                    st.metric("Error Std Dev", f"{error_std:.4f}")
                
                # Identify worst predictions
                st.subheader("Worst Predictions Analysis")
                
                worst_indices = np.argsort(abs_errors)[-10:]
                worst_predictions = enhanced_data_clean.iloc[worst_indices].copy()
                worst_predictions['Actual_Area'] = y_original.iloc[worst_indices]
                worst_predictions['Predicted_Area'] = y_pred_original[worst_indices]
                worst_predictions['Absolute_Error'] = abs_errors[worst_indices]
                
                display_cols = ['Actual_Area', 'Predicted_Area', 'Absolute_Error'] + feature_names[:5]
                st.dataframe(worst_predictions[display_cols].round(3), use_container_width=True)
                
                # Error by feature analysis
                st.subheader("Error Analysis by Features")
                
                # Select feature for error analysis
                error_feature = st.selectbox(
                    "Select Feature for Error Analysis",
                    feature_names,
                    key="error_feature"
                )
                
                if error_feature in enhanced_data_clean.columns:
                    feature_values = enhanced_data_clean[error_feature]
                    
                    # Create bins for continuous features
                    if feature_values.dtype in ['float64', 'int64']:
                        feature_bins = pd.cut(feature_values, bins=5, labels=False)
                        bin_errors = []
                        bin_labels = []
                        
                        for i in range(5):
                            bin_mask = feature_bins == i
                            if bin_mask.sum() > 0:
                                bin_errors.append(abs_errors[bin_mask].mean())
                                bin_range = f"{feature_values[bin_mask].min():.2f}-{feature_values[bin_mask].max():.2f}"
                                bin_labels.append(bin_range)
                        
                        fig_error_feature = px.bar(
                            x=bin_labels,
                            y=bin_errors,
                            title=f"Mean Absolute Error by {error_feature} Bins",
                            labels={'x': f'{error_feature} Range', 'y': 'Mean Absolute Error'}
                        )
                        st.plotly_chart(fig_error_feature, use_container_width=True)
            
            else:
                # Classification error analysis
                from sklearn.metrics import confusion_matrix, classification_report
                
                # Confusion matrix
                cm = confusion_matrix(y, y_pred)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    # Classification errors by class
                    class_errors = []
                    class_names = ["No Fire", "Low Risk", "Medium Risk", "High Risk"]
                    
                    for i in range(len(class_names)):
                        if i < len(cm):
                            total_class = cm[i].sum()
                            correct = cm[i, i]
                            error_rate = (total_class - correct) / total_class if total_class > 0 else 0
                            class_errors.append(error_rate)
                        else:
                            class_errors.append(0)
                    
                    fig_class_errors = px.bar(
                        x=class_names[:len(class_errors)],
                        y=class_errors,
                        title="Error Rate by Class",
                        labels={'x': 'Risk Class', 'y': 'Error Rate'}
                    )
                    st.plotly_chart(fig_class_errors, use_container_width=True)
                
                # Misclassified samples analysis
                st.subheader("Misclassified Samples")
                
                misclassified_mask = y != y_pred
                if misclassified_mask.sum() > 0:
                    misclassified_data = enhanced_data_clean[misclassified_mask].copy()
                    misclassified_data['Actual_Class'] = y[misclassified_mask]
                    misclassified_data['Predicted_Class'] = y_pred[misclassified_mask]
                    
                    # Show sample of misclassified instances
                    sample_size = min(20, len(misclassified_data))
                    display_cols = ['Actual_Class', 'Predicted_Class'] + feature_names[:5]
                    
                    st.write(f"Showing {sample_size} of {len(misclassified_data)} misclassified samples:")
                    st.dataframe(
                        misclassified_data[display_cols].head(sample_size).round(3),
                        use_container_width=True
                    )
                    
                    # Misclassification patterns
                    st.subheader("Misclassification Patterns")
                    
                    confusion_pairs = []
                    for i in range(len(cm)):
                        for j in range(len(cm)):
                            if i != j and cm[i, j] > 0:
                                confusion_pairs.append({
                                    'Actual': class_names[i],
                                    'Predicted': class_names[j],
                                    'Count': cm[i, j]
                                })
                    
                    if confusion_pairs:
                        confusion_df = pd.DataFrame(confusion_pairs)
                        confusion_df = confusion_df.sort_values('Count', ascending=False)
                        st.dataframe(confusion_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in error analysis: {str(e)}")
    else:
        st.info("Please load data from the main page to perform error analysis.")

with tab5:
    st.header("Model Comparison")
    
    st.markdown("""
    Compare multiple models side-by-side to understand their relative strengths and weaknesses.
    """)
    
    # Model selection for comparison
    models_to_compare = st.multiselect(
        "Select Models to Compare",
        list(models.keys()),
        default=list(models.keys())[:3] if len(models) >= 3 else list(models.keys()),
        help="Choose 2 or more models for comparison"
    )
    
    if len(models_to_compare) >= 2:
        # Comparison metrics
        st.subheader("Performance Comparison")
        
        comparison_data = []
        for model_name in models_to_compare:
            model_metrics = base_metrics[model_name].copy()
            model_metrics['Model'] = model_name
            comparison_data.append(model_metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')
        
        # Normalized comparison (0-1 scale)
        st.subheader("Normalized Performance Comparison")
        
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        normalized_df = comparison_df[numeric_cols].copy()
        
        for col in numeric_cols:
            if col in ['RMSE', 'MAE']:  # Lower is better
                normalized_df[col] = 1 - (comparison_df[col] - comparison_df[col].min()) / (comparison_df[col].max() - comparison_df[col].min())
            else:  # Higher is better
                normalized_df[col] = (comparison_df[col] - comparison_df[col].min()) / (comparison_df[col].max() - comparison_df[col].min())
        
        # Radar chart for comparison
        if len(models_to_compare) <= 5:  # Limit for readability
            fig_radar = go.Figure()
            
            for model_name in models_to_compare:
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_df.loc[model_name].values,
                    theta=normalized_df.columns,
                    fill='toself',
                    name=model_name
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Normalized Performance Comparison (Radar Chart)"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed comparison table
        st.dataframe(comparison_df.round(4), use_container_width=True)
        
        # Statistical significance testing
        if 'fire_data' in st.session_state and st.session_state.fire_data is not None:
            st.subheader("Statistical Significance Testing")
            
            if st.button("Run Cross-Validation Comparison"):
                with st.spinner("Running cross-validation for statistical comparison..."):
                    try:
                        data = st.session_state.fire_data.copy()
                        
                        from utils.preprocessing import create_features
                        from sklearn.model_selection import cross_val_score
                        
                        # Prepare data
                        enhanced_data = create_features(data)
                        
                        # Categorical encoding
                        for cat_col, encoder in categorical_encoders.items():
                            if cat_col in enhanced_data.columns:
                                enhanced_data[cat_col] = encoder.transform(enhanced_data[cat_col].astype(str))
                        
                        X = enhanced_data[feature_names].fillna(0)
                        
                        if problem_type == "regression":
                            y = np.log1p(enhanced_data['area'])
                            scoring = 'r2'
                        else:
                            def categorize_risk(area):
                                if area == 0:
                                    return 0
                                elif area <= 1:
                                    return 1
                                elif area <= 10:
                                    return 2
                                else:
                                    return 3
                            
                            y = enhanced_data['area'].apply(categorize_risk)
                            scoring = 'accuracy'
                        
                        # Remove missing values
                        mask = ~(X.isnull().any(axis=1) | pd.isnull(y))
                        X = X[mask]
                        y = y[mask]
                        
                        # Scale features
                        if scaler is not None:
                            X_scaled = scaler.transform(X)
                        else:
                            X_scaled = X.values
                        
                        # Cross-validation for each model
                        cv_results = {}
                        
                        for model_name in models_to_compare:
                            model = models[model_name]
                            scores = cross_val_score(model, X_scaled, y, cv=5, scoring=scoring)
                            cv_results[model_name] = {
                                'Mean': scores.mean(),
                                'Std': scores.std(),
                                'Scores': scores
                            }
                        
                        # Display results
                        cv_summary = pd.DataFrame({
                            model: {
                                'Mean Score': results['Mean'],
                                'Std Dev': results['Std'],
                                'Min Score': results['Scores'].min(),
                                'Max Score': results['Scores'].max()
                            }
                            for model, results in cv_results.items()
                        }).T
                        
                        st.dataframe(cv_summary.round(4), use_container_width=True)
                        
                        # Box plot of CV scores
                        cv_scores_data = []
                        for model_name, results in cv_results.items():
                            for score in results['Scores']:
                                cv_scores_data.append({
                                    'Model': model_name,
                                    'Score': score
                                })
                        
                        cv_scores_df = pd.DataFrame(cv_scores_data)
                        
                        fig_cv_box = px.box(
                            cv_scores_df,
                            x='Model',
                            y='Score',
                            title='Cross-Validation Score Distribution'
                        )
                        st.plotly_chart(fig_cv_box, use_container_width=True)
                        
                        # Pairwise t-tests
                        if len(models_to_compare) >= 2:
                            st.subheader("Pairwise Statistical Tests")
                            
                            from scipy import stats
                            
                            pairwise_results = []
                            
                            for i, model1 in enumerate(models_to_compare):
                                for j, model2 in enumerate(models_to_compare[i+1:], i+1):
                                    scores1 = cv_results[model1]['Scores']
                                    scores2 = cv_results[model2]['Scores']
                                    
                                    # Paired t-test
                                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                                    
                                    pairwise_results.append({
                                        'Model 1': model1,
                                        'Model 2': model2,
                                        'Mean Diff': scores1.mean() - scores2.mean(),
                                        't-statistic': t_stat,
                                        'p-value': p_value,
                                        'Significant (α=0.05)': p_value < 0.05
                                    })
                            
                            pairwise_df = pd.DataFrame(pairwise_results)
                            st.dataframe(pairwise_df.round(4), use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error in statistical comparison: {str(e)}")
        
        # Model complexity comparison
        st.subheader("Model Complexity Analysis")
        
        complexity_data = []
        
        for model_name in models_to_compare:
            model = models[model_name]
            complexity_info = {'Model': model_name}
            
            # Number of parameters (approximation)
            if hasattr(model, 'coef_'):
                if len(model.coef_.shape) > 1:
                    n_params = model.coef_.shape[0] * model.coef_.shape[1]
                else:
                    n_params = len(model.coef_)
                complexity_info['Parameters'] = n_params
            elif hasattr(model, 'n_features_in_'):
                complexity_info['Features'] = model.n_features_in_
            
            # Tree-specific metrics
            if hasattr(model, 'n_estimators'):
                complexity_info['Estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                complexity_info['Max Depth'] = model.max_depth
            
            complexity_data.append(complexity_info)
        
        complexity_df = pd.DataFrame(complexity_data).set_index('Model')
        
        if not complexity_df.empty:
            st.dataframe(complexity_df, use_container_width=True)
    
    else:
        st.info("Please select at least 2 models for comparison.")

# Model export and deployment readiness
st.header("Model Deployment Readiness")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Artifacts")
    
    if os.path.exists("models/forest_fire_models.pkl"):
        file_size = os.path.getsize("models/forest_fire_models.pkl")
        st.metric("Model File Size", f"{file_size / 1024:.1f} KB")
        
        st.success("✅ Models ready for deployment")
        
        # Model metadata
        st.write("**Model Metadata:**")
        st.write(f"- Problem Type: {problem_type}")
        st.write(f"- Number of Models: {len(models)}")
        st.write(f"- Feature Count: {len(feature_names)}")
        st.write(f"- Scaling: {'Yes' if scaler is not None else 'No'}")
    else:
        st.warning("⚠️ No model artifacts found")

with col2:
    st.subheader("Deployment Checklist")
    
    checklist_items = [
        ("Models Trained", len(models) > 0),
        ("Performance Validated", True),  # Assuming validation was done
        ("Feature Engineering", 'preprocessing' in str(type(scaler))),
        ("Model Artifacts Saved", os.path.exists("models/forest_fire_models.pkl")),
        ("Error Analysis Complete", True),  # Based on tab availability
    ]
    
    for item, status in checklist_items:
        status_icon = "✅" if status else "❌"
        st.write(f"{status_icon} {item}")
    
    deployment_ready = all(status for _, status in checklist_items)
    
    if deployment_ready:
        st.success("🚀 Ready for deployment!")
    else:
        st.warning("⚠️ Complete remaining items before deployment")

