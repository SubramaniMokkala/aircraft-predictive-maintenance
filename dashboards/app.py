import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import os

# Get base path for file loading
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# Title and description
st.title("✈️ Aircraft Predictive Maintenance System")
st.markdown("""
This system predicts the Remaining Useful Life (RUL) of aircraft turbofan engines using machine learning.
Upload sensor data or use the demo data to see predictions.
""")

# Sidebar
st.sidebar.header("About This Project")
st.sidebar.info("""
**Models Available:**
- Random Forest (Classical ML)
- LSTM Deep Learning

**Performance:**
- LSTM RMSE: ~25 cycles
- Random Forest RMSE: ~37 cycles

**Dataset:** NASA Turbofan Engine Degradation
""")

st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose prediction model:",
    ["LSTM (Recommended)", "Random Forest"]
)

# Load models
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load(os.path.join(BASE_PATH, 'models', 'random_forest_model.pkl'))
        lstm_model = keras.models.load_model(os.path.join(BASE_PATH, 'models', 'lstm_model.keras'))
        scaler = joblib.load(os.path.join(BASE_PATH, 'models', 'scaler.pkl'))
        lstm_config = joblib.load(os.path.join(BASE_PATH, 'models', 'lstm_config.pkl'))
        return rf_model, lstm_model, scaler, lstm_config
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

rf_model, lstm_model, scaler, lstm_config = load_models()

# Load demo data
@st.cache_data
def load_demo_data():
    # Column names
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]
    column_names = index_names + setting_names + sensor_names
    
    # Load data
    df = pd.read_csv(os.path.join(BASE_PATH, 'data', 'train_FD001.txt'), 
                     sep=' ', header=None, names=column_names, index_col=False)
    df = df.dropna(axis=1, how='all')
    
    # Calculate RUL
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycles'] - df['time_cycles']
    
    return df

demo_data = load_demo_data()

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Make Prediction", "📈 Model Performance", "ℹ️ About"])

with tab1:
    st.header("Predict Engine RUL")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Method")
        input_method = st.radio("Choose input:", ["Use Demo Data", "Manual Input"])
        
        if input_method == "Use Demo Data":
            engine_id = st.selectbox("Select Engine ID:", demo_data['unit_number'].unique())
            engine_data = demo_data[demo_data['unit_number'] == engine_id]
            
            st.write(f"**Engine {engine_id} Information:**")
            st.write(f"- Total cycles recorded: {len(engine_data)}")
            st.write(f"- Actual RUL: {engine_data.iloc[-1]['RUL']:.0f} cycles")
            
            # Get last reading
            last_reading = engine_data.iloc[-1]
            
        else:
            st.write("Enter sensor values (normalized 0-1):")
            sensor_inputs = {}
            for i in range(2, 22):
                if i not in [1, 5, 6, 10, 16, 18, 19]:  # Skip low-variance sensors
                    sensor_inputs[f'sensor_{i}'] = st.slider(
                        f'Sensor {i}', 0.0, 1.0, 0.5, 0.01
                    )
    
    with col2:
        st.subheader("Prediction Results")
        
        if st.button("🔮 Predict RUL", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    if input_method == "Use Demo Data":
                        # Prepare features
                        feature_cols = [col for col in engine_data.columns 
                                      if col.startswith('sensor_') or col.startswith('setting_')]
                        feature_cols = [col for col in feature_cols 
                                      if col not in ['sensor_1', 'sensor_5', 'sensor_6', 
                                                    'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']]
                        
                        if model_choice == "Random Forest":
                            # Create rolling features
                            engine_processed = engine_data.copy()
                            rolling_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 
                                             'sensor_8', 'sensor_11', 'sensor_12']
                            for sensor in rolling_sensors:
                                for window in [5, 10]:
                                    col_name = f'{sensor}_rolling_mean_{window}'
                                    engine_processed[col_name] = engine_processed[sensor].rolling(
                                        window=window, min_periods=1).mean()
                            
                            # Get all features
                            all_features = [col for col in engine_processed.columns 
                                          if col.startswith('sensor_') or col.startswith('setting_') 
                                          or 'rolling' in col]
                            all_features = [col for col in all_features 
                                          if col not in ['sensor_1', 'sensor_5', 'sensor_6', 
                                                        'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']]
                            
                            X = engine_processed[all_features].iloc[-1:].values
                            prediction = rf_model.predict(X)[0]
                            
                        else:  # LSTM
                            seq_length = lstm_config['sequence_length']
                            features = lstm_config['features']
                            
                            if len(engine_data) >= seq_length:
                                sequence = engine_data[features].iloc[-seq_length:].values
                                sequence = sequence.reshape(1, seq_length, len(features))
                                prediction = lstm_model.predict(sequence, verbose=0)[0][0]
                            else:
                                st.warning(f"Need at least {seq_length} cycles for LSTM prediction")
                                prediction = None
                        
                        if prediction is not None:
                            actual_rul = engine_data.iloc[-1]['RUL']
                            error = abs(actual_rul - prediction)
                            
                            st.success("✅ Prediction Complete!")
                            
                            # Display metrics
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            metric_col1.metric("Predicted RUL", f"{prediction:.1f} cycles")
                            metric_col2.metric("Actual RUL", f"{actual_rul:.0f} cycles")
                            metric_col3.metric("Error", f"{error:.1f} cycles")
                            
                            # Interpretation
                            st.write("**Interpretation:**")
                            if prediction < 30:
                                st.error("⚠️ Critical: Schedule maintenance immediately")
                            elif prediction < 60:
                                st.warning("⚡ Warning: Schedule maintenance soon")
                            else:
                                st.info("✓ Healthy: Continue normal operations")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")

with tab2:
    st.header("Model Performance Comparison")
    
    # Performance metrics
    metrics_df = pd.DataFrame({
        'Model': ['LSTM', 'Random Forest', 'Linear Regression'],
        'RMSE (cycles)': [25.00, 36.51, 43.38],
        'MAE (cycles)': [16.92, 25.51, 33.18],
        'R² Score': [0.8366, 0.7082, 0.5881]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        
        st.write("""
        **Why LSTM performs better:**
        - Captures temporal degradation patterns
        - Learns from 30-cycle sequences
        - Better handles sensor interactions
        """)
    
    with col2:
        st.subheader("RMSE Comparison")
        fig = px.bar(metrics_df, x='Model', y='RMSE (cycles)', 
                     color='Model', text='RMSE (cycles)')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("About This System")
    
    st.markdown("""
    ### Project Overview
    This predictive maintenance system uses machine learning to forecast when aircraft engines 
    will fail, allowing airlines to schedule maintenance proactively.
    
    ### Dataset
    - **Source:** NASA Turbofan Engine Degradation Simulation
    - **Engines:** 100 run-to-failure samples
    - **Sensors:** 21 different measurements (temperature, pressure, vibration, etc.)
    - **Target:** Remaining Useful Life (RUL) in flight cycles
    
    ### Models
    
    **1. Random Forest (Classical ML)**
    - Ensemble of 100 decision trees
    - Uses hand-crafted rolling window features
    - RMSE: 36.51 cycles
    
    **2. LSTM Deep Learning (Recommended)**
    - Recurrent neural network with memory
    - Learns temporal patterns automatically
    - RMSE: 25.00 cycles (31% better)
    
    ### Business Impact
    - Average prediction error: ~25 cycles (2-3 weeks)
    - Enables proactive maintenance scheduling
    - Reduces unscheduled downtime
    - Improves flight safety and operational efficiency
    
    ### Technical Stack
    - Python, TensorFlow/Keras, Scikit-learn
    - Streamlit for dashboard
    - Plotly for visualizations
    
    ---
    
    **Created for Airbus AI/ML Engineer Internship Application**
    """)

# Footer
st.markdown("---")
st.markdown("📧 Contact: Subramani Mokkala | 🔗 GitHub: [aircraft-predictive-maintenance](https://github.com/SubramaniMokkala/aircraft-predictive-maintenance)")