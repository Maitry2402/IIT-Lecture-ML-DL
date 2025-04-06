import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set page title and layout
st.set_page_config(page_title="Cyberattack Classification", layout="wide")

# Title and description
st.title("Cyberattack Classification Dashboard")
st.markdown("This application demonstrates the classification of cyberattacks using various machine learning and deep learning models.")

# Load models
@st.cache_resource
def load_models():
    return {}

# Load dataset
@st.cache_data
def load_data():
    try:
        # Update the path to your CSV file
        df = pd.read_csv("rt-iot22.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load top features
@st.cache_data
def load_top_features():
    top_features = {}
    feature_files = {
        'xgboost': 'top_features_xgboost.txt',
        'random_forest': 'top_features_random_forest.txt',
        'lightgbm': 'top_features_lightgbm.txt'
    }
    
    for model_name, file_path in feature_files.items():
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    top_features[model_name] = [line.strip() for line in f.readlines()]
            else:
                top_features[model_name] = []
        except Exception as e:
            top_features[model_name] = []
    
    return top_features

# Function to load saved images
def load_image(image_path):
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            st.warning(f"Image file {image_path} not found.")
            return None
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

# Main function
def main():
    # Load models, data, and top features
    models = load_models()
    df = load_data()
    top_features = load_top_features()
    
    if df is None:
        st.error("Failed to load dataset. Please check the file path.")
        return
    
    # Create tabs for ML and DL models
    tab_ml, tab_dl = st.tabs(["Machine Learning Models", "Deep Learning Models"])
    
    # ML MODELS TAB
    with tab_ml:
        st.header("Machine Learning Models")
        st.write("Analysis of XGBoost, Random Forest, and LightGBM models")
        
        # Display dataset information
        st.header("Dataset Overview")
        st.write(f"Dataset Shape: {df.shape}")
        
        # Sidebar for user interaction - ML part
        with st.sidebar:
            st.header("Machine Learning Options")
            
            # Select model type
            ml_model_type = st.selectbox(
                "Select ML Model Type",
                options=["XGBoost", "Random Forest", "LightGBM"],
                key="ml_model"
            )
            
            # Map selected model type to model keys
            ml_model_type_lower = ml_model_type.lower().replace(" ", "_")
            
            # Get all attack types
            attack_types = df['Attack_type'].unique().tolist()
            
            # User selects attack type
            ml_selected_attack = st.selectbox(
                "Select Attack Type to Analyze",
                options=attack_types,
                key="ml_attack"
            )
        
        # Filter data for selected attack type
        ml_filtered_data = df[df['Attack_type'] == ml_selected_attack]
        
        # Display information about selected attack type
        st.header(f"Analysis for Attack Type: {ml_selected_attack}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Data")
            st.write(ml_filtered_data.head())
            
            st.subheader("Count")
            st.write(f"Number of instances: {len(ml_filtered_data)}")
        
        with col2:
            # Display feature statistics
            st.subheader("Feature Statistics")
            st.write(ml_filtered_data.describe())
        
        # Model Performance Metrics
        st.header(f"Model Performance Visualization - {ml_model_type}")
        
        # Create tabs for different visuals
        ml_tab1, ml_tab2, ml_tab3, ml_tab4, ml_tab5 = st.tabs(["Confusion Matrix", "ROC Curve", "PR Curve", "Feature Importance", "LIME Interpretation"])
        
        with ml_tab1:
            st.subheader("Confusion Matrix")
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("Full Model")
                cm_full = load_image(f"confusion_matrix_{ml_model_type_lower}_full_model.png")
                if cm_full:
                    st.image(cm_full, caption=f"Confusion Matrix - {ml_model_type} Full Model", use_container_width=True)
                else:
                    st.info(f"Confusion matrix image for {ml_model_type} full model not found.")
                
                # Add classification report for full model
                if ml_model_type == "XGBoost":
                    st.subheader("Classification Report - XGBoost Full Model")
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                        'Value': [0.99, 0.97, 0.95, 0.96]
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    st.dataframe(df_metrics, hide_index=True)
                elif ml_model_type == "Random Forest":
                    st.subheader("Classification Report - Random Forest Full Model")
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                        'Value': [0.99, 0.96, 0.96, 0.96]
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    st.dataframe(df_metrics, hide_index=True)
                elif ml_model_type == "LightGBM":
                    st.subheader("Classification Report - LightGBM Full Model")
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                        'Value': [1.00, 0.98, 0.95, 0.97]
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    st.dataframe(df_metrics, hide_index=True)
            
            with col2:
                st.caption("Top 10 Features Model")
                cm_top10 = load_image(f"confusion_matrix_{ml_model_type_lower}_top10_model.png")
                if cm_top10:
                    st.image(cm_top10, caption=f"Confusion Matrix - {ml_model_type} Top 10 Features Model", use_container_width=True)
                else:
                    st.info(f"Confusion matrix image for {ml_model_type} top10 model not found.")
                
                # Add classification report for top10 model
                if ml_model_type == "XGBoost":
                    st.subheader("Classification Report - XGBoost Top10 Model")
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                        'Value': [0.99, 0.90, 0.94, 0.92]
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    st.dataframe(df_metrics, hide_index=True)
                elif ml_model_type == "Random Forest":
                    st.subheader("Classification Report - Random Forest Top10 Model")
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                        'Value': [0.96, 0.71, 0.80, 0.73]
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    st.dataframe(df_metrics, hide_index=True)
                elif ml_model_type == "LightGBM":
                    st.subheader("Classification Report - LightGBM Top10 Model")
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                        'Value': [0.98, 0.92, 0.91, 0.91]
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    st.dataframe(df_metrics, hide_index=True)
        
        with ml_tab2:
            st.subheader("ROC Curve")
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("Full Model")
                roc_full = load_image(f"roc_curve_{ml_model_type_lower}_full_model.png")
                if roc_full:
                    st.image(roc_full, caption=f"ROC Curve - {ml_model_type} Full Model", use_container_width=True)
                else:
                    st.info(f"ROC curve image for {ml_model_type} full model not found.")
            
            with col2:
                st.caption("Top 10 Features Model")
                roc_top10 = load_image(f"roc_curve_{ml_model_type_lower}_top10_model.png")
                if roc_top10:
                    st.image(roc_top10, caption=f"ROC Curve - {ml_model_type} Top 10 Features Model", use_container_width=True)
                else:
                    st.info(f"ROC curve image for {ml_model_type} top10 model not found.")
        
        with ml_tab3:
            st.subheader("Precision-Recall Curve")
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("Full Model")
                pr_full = load_image(f"pr_curve_{ml_model_type_lower}_full_model.png")
                if pr_full:
                    st.image(pr_full, caption=f"PR Curve - {ml_model_type} Full Model", use_container_width=True)
                else:
                    st.info(f"PR curve image for {ml_model_type} full model not found.")
            
            with col2:
                st.caption("Top 10 Features Model")
                pr_top10 = load_image(f"pr_curve_{ml_model_type_lower}_top10_model.png")
                if pr_top10:
                    st.image(pr_top10, caption=f"PR Curve - {ml_model_type} Top 10 Features Model", use_container_width=True)
                else:
                    st.info(f"PR curve image for {ml_model_type} top10 model not found.")
        
        with ml_tab4:
            st.subheader("Feature Importance Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                shap_fi = load_image(f"shap_feature_importance_{ml_model_type_lower}.png")
                if shap_fi:
                    st.image(shap_fi, caption=f"SHAP Feature Importance - {ml_model_type}", use_container_width=True)
                else:
                    st.info(f"SHAP feature importance image for {ml_model_type} not found.")
            
            with col2:
                shap_summary = load_image(f"shap_summary_plot_{ml_model_type_lower}.png")
                if shap_summary:
                    st.image(shap_summary, caption=f"SHAP Summary Plot - {ml_model_type}", use_container_width=True)
                else:
                    st.info(f"SHAP summary plot image for {ml_model_type} not found.")
            
            # Display top 10 features for the selected model
            if ml_model_type_lower in top_features and top_features[ml_model_type_lower]:
                st.subheader(f"Top 10 Features - {ml_model_type}")
                for i, feature in enumerate(top_features[ml_model_type_lower], 1):
                    st.write(f"{i}. {feature}")
            else:
                st.info(f"Top features list for {ml_model_type} not found.")
        
        with ml_tab5:
            st.subheader("LIME Interpretation")
            st.write("LIME (Local Interpretable Model-agnostic Explanations) helps explain individual predictions.")
            
            # Create radio buttons for normal vs attack samples
            ml_interpretation_type = st.radio(
                "Select sample type for LIME interpretation",
                options=["Normal Traffic", "Attack Traffic"],
                horizontal=True,
                key="ml_lime"
            )
            
            # Load appropriate LIME images based on selection
            if ml_interpretation_type == "Normal Traffic":
                lime_img = load_image(f"lime_normal_{ml_model_type_lower}.png")
                if lime_img:
                    st.image(lime_img, caption=f"LIME Interpretation for Normal Traffic - {ml_model_type}", use_container_width=True)
                else:
                    st.info(f"LIME interpretation image for normal traffic with {ml_model_type} not found.")
            else:  # Attack Traffic
                lime_img = load_image(f"lime_attack_{ml_model_type_lower}.png")
                if lime_img:
                    st.image(lime_img, caption=f"LIME Interpretation for Attack Traffic - {ml_model_type}", use_container_width=True)
                else:
                    st.info(f"LIME interpretation image for attack traffic with {ml_model_type} not found.")
        
        # Model Comparison
        st.header("ML Model Comparison")
        
        # Create comparison tabs
        comp_tab1, comp_tab2, comp_tab3 = st.tabs(["All Full Models", "All Top10 Models", "Full vs Top10"])
        
        with comp_tab1:
            st.subheader("Comparison of All Full Models")
            
            # Example: Compare confusion matrices
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cm_xgb = load_image("confusion_matrix_xgboost_full_model.png")
                if cm_xgb:
                    st.image(cm_xgb, caption="XGBoost Full Model", use_container_width=True)
                else:
                    st.info("XGBoost full model confusion matrix not found.")
            
            with col2:
                cm_rf = load_image("confusion_matrix_random_forest_full_model.png")
                if cm_rf:
                    st.image(cm_rf, caption="Random Forest Full Model", use_container_width=True)
                else:
                    st.info("Random Forest full model confusion matrix not found.")
            
            with col3:
                cm_lgb = load_image("confusion_matrix_lightgbm_full_model.png")
                if cm_lgb:
                    st.image(cm_lgb, caption="LightGBM Full Model", use_container_width=True)
                else:
                    st.info("LightGBM full model confusion matrix not found.")
        
        with comp_tab2:
            st.subheader("Comparison of All Top10 Models")
            
            # Example: Compare confusion matrices
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cm_xgb = load_image("confusion_matrix_xgboost_top10_model.png")
                if cm_xgb:
                    st.image(cm_xgb, caption="XGBoost Top10 Model", use_container_width=True)
                else:
                    st.info("XGBoost top10 model confusion matrix not found.")
            
            with col2:
                cm_rf = load_image("confusion_matrix_random_forest_top10_model.png")
                if cm_rf:
                    st.image(cm_rf, caption="Random Forest Top10 Model", use_container_width=True)
                else:
                    st.info("Random Forest top10 model confusion matrix not found.")
            
            with col3:
                cm_lgb = load_image("confusion_matrix_lightgbm_top10_model.png")
                if cm_lgb:
                    st.image(cm_lgb, caption="LightGBM Top10 Model", use_container_width=True)
                else:
                    st.info("LightGBM top10 model confusion matrix not found.")
        
        with comp_tab3:
            st.subheader("Full Model vs Top10 Model Comparison")
            st.write("Select a model type to compare its full version against the top10 features version.")
            
            # Dropdown to select model for comparison
            comp_model = st.selectbox(
                "Select Model for Comparison",
                options=["XGBoost", "Random Forest", "LightGBM"],
                key="comp_model"
            )
            
            comp_model_lower = comp_model.lower().replace(" ", "_")
            
            # Compare ROC curves
            st.subheader(f"ROC Curve Comparison - {comp_model}")
            col1, col2 = st.columns(2)
            
            with col1:
                roc_full = load_image(f"roc_curve_{comp_model_lower}_full_model.png")
                if roc_full:
                    st.image(roc_full, caption=f"{comp_model} Full Model", use_container_width=True)
                else:
                    st.info(f"{comp_model} full model ROC curve not found.")
            
            with col2:
                roc_top10 = load_image(f"roc_curve_{comp_model_lower}_top10_model.png")
                if roc_top10:
                    st.image(roc_top10, caption=f"{comp_model} Top10 Model", use_container_width=True)
                else:
                    st.info(f"{comp_model} top10 model ROC curve not found.")
    
    # DL MODELS TAB
    with tab_dl:
        st.header("Deep Learning Models")
        st.write("Analysis of MLP, 1D-CNN, and TabNet-like models for cyberattack detection")
        
        # Display dataset information
        st.header("Dataset Overview")
        st.write(f"Dataset Shape: {df.shape}")
        
        # Sidebar for user interaction - DL part
        with st.sidebar:
            st.header("Deep Learning Options")
            
            # Select model type
            dl_model_type = st.selectbox(
                "Select DL Model Type",
                options=["MLP", "1D-CNN", "TabNet-like"],
                key="dl_model"
            )
            
            # Map selected model type to model keys
            dl_model_type_lower = dl_model_type.lower().replace("-", "").replace(" ", "_")
            
            # Get all attack types
            attack_types = df['Attack_type'].unique().tolist()
            
            # User selects attack type
            dl_selected_attack = st.selectbox(
                "Select Attack Type to Analyze",
                options=attack_types,
                key="dl_attack"
            )
        
        # Filter data for selected attack type
        dl_filtered_data = df[df['Attack_type'] == dl_selected_attack]
        
        # Display information about selected attack type
        st.header(f"Analysis for Attack Type: {dl_selected_attack}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Data")
            st.write(dl_filtered_data.head())
            
            st.subheader("Count")
            st.write(f"Number of instances: {len(dl_filtered_data)}")
        
        with col2:
            # Display feature statistics
            st.subheader("Feature Statistics")
            st.write(dl_filtered_data.describe())
        
        # Model Performance Metrics
        st.header(f"Model Performance Visualization - {dl_model_type}")
        
        # Create tabs for different visuals
        dl_tab1, dl_tab2, dl_tab3, dl_tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "PR Curve", "LIME Interpretation"])
        
        with dl_tab1:
            st.subheader("Confusion Matrix")
            cm_dl = load_image(f"confusion_matrix_{dl_model_type_lower}_full_model.png")
            if cm_dl:
                st.image(cm_dl, caption=f"Confusion Matrix - {dl_model_type}", use_container_width=True)
            else:
                st.info(f"Confusion matrix image for {dl_model_type} not found.")
            
            # Add classification report for DL models
            if dl_model_type == "1D-CNN":
                st.subheader("Classification Report - 1D CNN Full Model")
                metrics_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                    'Value': [0.99, 0.86, 0.96, 0.90]
                }
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, hide_index=True)
            elif dl_model_type == "MLP":
                st.subheader("Classification Report - MLP Full Model")
                metrics_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                    'Value': [0.98, 0.84, 0.91, 0.84]
                }
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, hide_index=True)
            elif dl_model_type == "TabNet-like":
                st.subheader("Classification Report - TabNet-like Full Model")
                metrics_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                    'Value': [0.99, 0.94, 0.97, 0.95]
                }
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, hide_index=True)
        
        with dl_tab2:
            st.subheader("ROC Curve")
            roc_dl = load_image(f"roc_curve_{dl_model_type_lower}_full_model.png")
            if roc_dl:
                st.image(roc_dl, caption=f"ROC Curve - {dl_model_type}", use_container_width=True)
            else:
                st.info(f"ROC curve image for {dl_model_type} not found.")
        
        with dl_tab3:
            st.subheader("Precision-Recall Curve")
            pr_dl = load_image(f"pr_curve_{dl_model_type_lower}_full_model.png")
            if pr_dl:
                st.image(pr_dl, caption=f"PR Curve - {dl_model_type}", use_container_width=True)
            else:
                st.info(f"PR curve image for {dl_model_type} not found.")
        
        with dl_tab4:
            st.subheader("LIME Interpretation")
            st.write("LIME (Local Interpretable Model-agnostic Explanations) helps explain individual predictions.")
            
            # Create radio buttons for normal vs attack samples
            dl_interpretation_type = st.radio(
                "Select sample type for LIME interpretation",
                options=["Normal Traffic", "Attack Traffic"],
                horizontal=True,
                key="dl_lime"
            )
            
            # Load appropriate LIME images based on selection
            if dl_interpretation_type == "Normal Traffic":
                lime_img = load_image(f"lime_normal_{dl_model_type_lower}.png")
                if lime_img:
                    st.image(lime_img, caption=f"LIME Interpretation for Normal Traffic - {dl_model_type}", use_container_width=True)
                else:
                    st.info(f"LIME interpretation image for normal traffic with {dl_model_type} not found.")
            else:  # Attack Traffic
                lime_img = load_image(f"lime_attack_{dl_model_type_lower}.png")
                if lime_img:
                    st.image(lime_img, caption=f"LIME Interpretation for Attack Traffic - {dl_model_type}", use_container_width=True)
                else:
                    st.info(f"LIME interpretation image for attack traffic with {dl_model_type} not found.")
            
            st.write("""
            ### About LIME Interpretation
            
            LIME explains individual predictions by approximating the model locally with an interpretable model.
            
            The colored bars show how each feature contributes to pushing the prediction toward (green) or away from (red) the predicted class.
            """)
        
        # Deep Learning Model Comparison
        st.header("DL Model Comparison")
        st.subheader("Comparison of Deep Learning Models")
        
        # Compare confusion matrices
        st.write("Confusion Matrix Comparison")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cm_mlp = load_image("confusion_matrix_mlp_full_model.png")
            if cm_mlp:
                st.image(cm_mlp, caption="MLP Model", use_container_width=True)
            else:
                st.info("MLP confusion matrix not found.")
        
        with col2:
            cm_cnn = load_image("confusion_matrix_1dcnn_full_model.png")
            if cm_cnn:
                st.image(cm_cnn, caption="1D-CNN Model", use_container_width=True)
            else:
                st.info("1D-CNN confusion matrix not found.")
        
        with col3:
            cm_tabnet = load_image("confusion_matrix_tabnetlike_full_model.png")
            if cm_tabnet:
                st.image(cm_tabnet, caption="TabNet-like Model", use_container_width=True)
            else:
                st.info("TabNet-like confusion matrix not found.")
        
        # Model architecture information
        st.subheader("Deep Learning Model Architectures")
        
        model_info_tab1, model_info_tab2, model_info_tab3 = st.tabs(["MLP", "1D-CNN", "TabNet-like"])
        
        with model_info_tab1:
            st.write("""
            ### Multi-Layer Perceptron (MLP)
            
            A standard feedforward neural network with:
            - 3 hidden layers (256, 128, 64 neurons)
            - ReLU activation functions
            - Dropout for regularization
            - L2 regularization on weights
            
            **Strengths:**
            - Simple architecture with good performance
            - Fast training and inference time
            - Good at handling tabular data
            
            **Limitations:**
            - May not capture complex patterns as effectively as CNN or TabNet
            - Requires careful tuning of architecture
            """)
        
        with model_info_tab2:
            st.write("""
            ### 1D Convolutional Neural Network (1D-CNN)
            
            A convolutional network architecture with:
            - Multiple Conv1D layers for feature extraction
            - MaxPooling layers for dimension reduction
            - Dropout for regularization
            - Dense layers for final classification
            
            **Strengths:**
            - Captures spatial patterns in features
            - Effective at learning local feature relationships
            - Good at handling sequential patterns
            
            **Limitations:**
            - More complex architecture than MLP
            - Requires reshaping of input data
            - May overfit on small datasets
            """)
        
        with model_info_tab3:
            st.write("""
            ### TabNet-like Model
            
            A simplified version of TabNet with:
            - Feature selection through attention mechanisms
            - Gated Linear Units (GLU) for feature transformation
            - Sequential decision steps for processing features
            - Skip connections between layers
            
            **Strengths:**
            - Feature selection through attention mechanism
            - Interpretable feature importance
            - Comparable performance to deep neural networks
            
            **Limitations:**
            - More complex training process
            - Less established than traditional neural networks
            - Custom implementation may not match official TabNet performance
            """)

# Run the application
if __name__ == "__main__":
    main()
