import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import streamlit.components.v1 as components

# --- Helper to display SHAP plots in Streamlit ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- Caching functions ---
@st.cache_data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    if 'Class' not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'Class' column (fraud labels).")
    X = df.drop(columns=['Class'])
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, y

@st.cache_data
def load_default_data():
    return load_and_preprocess_data("creditcard.csv")

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Page Config ---
st.set_page_config(page_title="Credit Card Fraud XAI", layout="wide")
st.title("💳 Credit Card Fraud Detection with Explainable AI")

# --- Tabs for UI ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Upload Data",
    "2. Train Model",
    "3. Explain with SHAP",
    "4. Download Reports",
    "5. Global Feature Importance"
])

# --- Shared state ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None

# --- Tab 1: Upload ---
with tab1:
    st.header("📁 Upload Credit Card Dataset")
    uploaded_file = st.file_uploader("Upload CSV with a 'Class' column", type="csv")
    if uploaded_file:
        try:
            X, y = load_and_preprocess_data(uploaded_file)
            st.session_state.X = X
            st.session_state.y = y
            st.success("✅ Data uploaded and preprocessed successfully.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("Using default dataset.")
        X, y = load_default_data()
        st.session_state.X = X
        st.session_state.y = y

# --- Tab 2: Train Model ---
with tab2:
    st.header("⚙️ Train Random Forest Classifier")
    if 'X' in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.X, st.session_state.y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.explainer = shap.Explainer(model, X_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("📊 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("📉 Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("🏑 ROC AUC Score")
        st.write(f"{roc_auc_score(y_test, y_proba):.4f}")

        st.subheader("📈 Fraud Probability Histogram")
        plt.figure(figsize=(10, 6))
        plt.hist(y_proba, bins=50, color='purple', alpha=0.7)
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.warning("⚠️ Please upload data in Tab 1.")

# --- Tab 3: SHAP Explanation ---
# --- Tab 3: SHAP Explanation ---
# --- Tab 3: SHAP Explanation (with Debug Mode) ---
# --- Tab 3: SHAP Explanation (with Debug Mode) ---
with tab3:
    st.header("🔍 Explain Model Predictions with SHAP")

    # Add debug toggle at the top
    debug_mode = st.checkbox("Enable Debug Mode", help="Show raw SHAP values and data structures")

    if st.session_state.model and st.session_state.X_test is not None:
        transaction_index = st.number_input(
            "Choose a transaction index to explain:",
            min_value=0,
            max_value=len(st.session_state.X_test) - 1,
            value=5
        )
        transaction = st.session_state.X_test.iloc[[transaction_index]]
        explainer = st.session_state.explainer

        try:
            # Generate SHAP values
            shap_values = explainer(transaction)

            if debug_mode:
                st.subheader("🔧 Debug Information")
                with st.expander("Raw SHAP Values Structure"):
                    st.write(f"Shape: {shap_values.shape}")
                    st.write(f"Type: {type(shap_values)}")

                    # Properly display SHAP values for both classes
                    st.write("Class 0 SHAP values:", pd.DataFrame(
                        shap_values.values[0, :, 0],
                        index=transaction.columns,
                        columns=["SHAP Value"]
                    ))
                    st.write("Class 1 SHAP values:", pd.DataFrame(
                        shap_values.values[0, :, 1],
                        index=transaction.columns,
                        columns=["SHAP Value"]
                    ))

                with st.expander("Transaction Data"):
                    st.write(transaction)

            # Handle both single-output and multi-output cases
            if len(shap_values.shape) == 3:  # Multi-output (classification)
                shap_values_display = shap_values[..., 1]  # Use class 1 (fraud) values
                expected_value = explainer.expected_value[1]
                if debug_mode:
                    st.write("Multi-output detected - Using class 1 values")
                    st.write(f"Expected value for class 1: {expected_value}")
            else:  # Single output
                shap_values_display = shap_values
                expected_value = explainer.expected_value
                if debug_mode:
                    st.write("Single output detected")
                    st.write(f"Expected value: {expected_value}")

            # Create consistent Explanation object
            explanation = shap.Explanation(
                values=shap_values_display[0],
                base_values=expected_value,
                data=transaction.iloc[0].values,
                feature_names=transaction.columns.tolist()
            )

            if debug_mode:
                with st.expander("Explanation Object Details"):
                    st.write("Values:", pd.DataFrame(
                        explanation.values,
                        index=transaction.columns,
                        columns=["SHAP Value"]
                    ))
                    st.write("Base values:", explanation.base_values)
                    st.write("Feature names:", explanation.feature_names)
                    st.write("Data:", pd.DataFrame(
                        explanation.data,
                        index=transaction.columns,
                        columns=["Feature Value"]
                    ))

            st.markdown("### 📈 Force Plot")
            try:
                if debug_mode:
                    st.write("Generating force plot with explanation object")

                force_plot = shap.plots.force(explanation)
                st_shap(force_plot, height=400)
            except Exception as e:
                st.error(f"Force plot error: {str(e)}")
                if debug_mode:
                    st.exception(e)

            st.markdown("### 🌊 Waterfall Plot")
            try:
                if debug_mode:
                    st.write("Generating waterfall plot with explanation object")

                plt.figure()
                shap.plots.waterfall(explanation, max_display=10)
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception as e:
                st.error(f"Waterfall plot error: {str(e)}")
                if debug_mode:
                    st.exception(e)

            st.markdown("### 📊 Top 5 Global Features")
            try:
                if debug_mode:
                    st.write("Calculating global feature importance...")

                X_test_sample = st.session_state.X_test.sample(n=min(1000, len(st.session_state.X_test)),
                                                               random_state=42)
                shap_vals_all = explainer(X_test_sample)

                if debug_mode:
                    st.write(f"SHAP values shape for global features: {shap_vals_all.shape}")

                # Handle multi-output case
                if len(shap_vals_all.shape) == 3:
                    mean_vals = np.abs(shap_vals_all.values[..., 1]).mean(axis=0)
                    if debug_mode:
                        st.write("Using class 1 values for global importance")
                else:
                    mean_vals = np.abs(shap_vals_all.values).mean(axis=0)
                #Rijowan khan 
                top_indices = np.argsort(mean_vals)[-5:][::-1]
                st.write(pd.DataFrame({
                    "Feature": st.session_state.X_test.columns[top_indices],
                    "Importance": mean_vals[top_indices]
                }))
            except Exception as e:
                st.error(f"Global features error: {str(e)}")
                if debug_mode:
                    st.exception(e)

        except Exception as e:
            st.error(f"SHAP explanation failed: {str(e)}")
            if debug_mode:
                st.exception(e)
    else:
        st.warning("⚠️ Please train the model first in Tab 2.")

# --- Tab 4: Download Reports ---
with tab4:
    st.header("📥 Download Processed Test Data & Predictions")
    if st.session_state.X_test is not None:
        model = st.session_state.model
        X_test = st.session_state.X_test.copy()
        y_test = st.session_state.y_test
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        result_df = X_test.copy()
        result_df["True_Label"] = y_test.values
        result_df["Predicted_Label"] = y_pred
        result_df["Fraud_Probability"] = y_proba

        csv = convert_df(result_df)
        st.download_button("📄 Download Predictions CSV",
                           data=csv,
                           file_name="fraud_predictions.csv",
                           mime="text/csv")
    else:
        st.warning("⚠️ No test data available. Train model in Tab 2.")

# --- Tab 5: Global Feature Importance ---
with tab5:
    st.header("📊 Global Feature Importance (Summary Plot)")
    if st.session_state.model and st.session_state.X_test is not None:
        X_sample = st.session_state.X_test.sample(n=min(500, len(st.session_state.X_test)), random_state=42)
        shap_values = st.session_state.explainer(X_sample)

        try:
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error creating summary plot: {str(e)}")
    else:
        st.warning("⚠️ Train the model in Tab 2 to view summary plot.")

# #author 👨‍💻 Author
# Md Rijowan Khan
# 🎓 B.Sc. in Computer Science & Technology
# 📍 China University of Petroleum (Beijing)
# 📧 Email: rijowan@qq.com
