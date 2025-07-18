import streamlit as st
import pandas as pd
import pickle
import time
import numpy as np
import os

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Load model ---
@st.cache_resource
def load_model():
    with open(r"C:\Users\srich\Documents\_Final Project Demo\seizure_detection_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- Session state initialization ---
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "simulation_started" not in st.session_state:
    st.session_state.simulation_started = False

# Features used by the ML model
significant_features = [
    'EEG RESP1-REF_hjorth_mobility',
    'EEG RESP2-REF_hjorth_mobility',
    'EEG EKG-REF_min'
]

# Features to display in the chart
display_features = [
    'Heart_Rate',
    'Resp_BPM'
]

if "features_history" not in st.session_state:
    st.session_state.features_history = pd.DataFrame(columns=display_features + ['time_seconds'])

if "dots_count" not in st.session_state:
    st.session_state.dots_count = 0

if "tval" not in st.session_state:
    st.session_state.tval = 0

# --- NEW: Seizure state tracking ---
if "seizure_active" not in st.session_state:
    st.session_state.seizure_active = False

if "consecutive_zeros" not in st.session_state:
    st.session_state.consecutive_zeros = 0

# --- Function to calculate percentage change and determine indicator ---
def calculate_indicator(df, current_idx):
    """
    Calculate percentage change between current and first row for Heart Rate and Respiratory Rate.
    Returns the indicator based on which vital sign shows the most drastic change.
    """
    if current_idx == 0:
        return ""  # First row, no comparison needed
    
    current_row = df.iloc[current_idx]
    first_row = df.iloc[0]  # Compare against first row instead of previous
    
    # Calculate percentage changes
    hr_change = 0
    resp_change = 0
    
    if 'Heart_Rate' in df.columns and first_row['Heart_Rate'] != 0:
        hr_change = abs((current_row['Heart_Rate'] - first_row['Heart_Rate']) / first_row['Heart_Rate'] * 100)
    
    if 'Resp_BPM' in df.columns and first_row['Resp_BPM'] != 0:
        resp_change = abs((current_row['Resp_BPM'] - first_row['Resp_BPM']) / first_row['Resp_BPM'] * 100)
    
    # Define threshold for "drastic change" (you can adjust this)
    threshold = 10.0  # 10% change threshold
    print(hr_change, resp_change, threshold)
    # Determine indicator based on most significant change
    if hr_change >= threshold and resp_change >= threshold:
        # Both changed significantly, return the one with larger change
        return "Heart Rate" if hr_change > resp_change else "Respiratory Rate"
    elif hr_change >= threshold:
        return "Heart Rate"
    elif resp_change >= threshold:
        return "Respiratory Rate"
    else:
        return ""  # No significant change

# --- NEW: Function to manage seizure state ---
def update_seizure_state(prediction):
    """
    Update seizure state based on prediction and consecutive zero count.
    Returns the display state (True for seizure display, False for normal).
    """
    if prediction == 1:
        # Seizure detected - activate seizure state and reset counter
        st.session_state.seizure_active = True
        st.session_state.consecutive_zeros = 0
        return True
    else:
        # No seizure detected
        if st.session_state.seizure_active:
            # We're in seizure state, count consecutive zeros
            st.session_state.consecutive_zeros += 1
            if st.session_state.consecutive_zeros > 3:
                # 3 consecutive zeros - clear seizure state
                st.session_state.seizure_active = False
                st.session_state.consecutive_zeros = 0
                return False
            else:
                # Still in seizure state, less than 3 consecutive zeros
                return True
        else:
            # Not in seizure state
            return False

# --- Upload CSV ---
if "uploaded_df" not in st.session_state or st.session_state.uploaded_df is None:
    st.title("🧠 Seizure Detection Simulation - File Upload")
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Save to a temporary path for overwriting during simulation
        temp_path = os.path.join("uploaded_temp.csv")
        df.to_csv(temp_path, index=False)

        st.session_state.uploaded_df = df
        st.session_state.uploaded_path = temp_path
        st.session_state.current_index = 0
        st.session_state.features_history = pd.DataFrame(columns=display_features + ['time_seconds'])
        st.session_state.tval = 0
        # Reset seizure state
        st.session_state.seizure_active = False
        st.session_state.consecutive_zeros = 0
        st.session_state.simulation_started = False

        # Just show success message and rerun to go to preview stage
        st.success("✅ File uploaded successfully!")
        st.rerun()

# --- File uploaded but simulation not started ---
elif not st.session_state.simulation_started:
    st.title("🧠 EEG Seizure Detection - Ready to Start")
    df = st.session_state.uploaded_df
    
    # Display file info
    st.info("📁 File is loaded and ready for simulation!")
    
    # Display file preview
    st.markdown("### 📄 File Preview")
    st.dataframe(df)

    # Add Indicator column if it doesn't exist
    if 'Indicator' not in df.columns:
        df['Indicator'] = ""
    
    # Display file statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        if hasattr(st.session_state, 'uploaded_path') and os.path.exists(st.session_state.uploaded_path):
            file_size = os.path.getsize(st.session_state.uploaded_path)
            st.metric("File Size", f"{file_size / 1024:.2f} KB")
    
    # Start simulation button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Simulation", type="primary", use_container_width=True):
            st.session_state.simulation_started = True
            # Clear the page by rerunning
            st.empty()
            st.rerun()

# --- Simulation (Completely clean monitoring screen) ---
elif st.session_state.simulation_started:
    # Only show monitoring interface, no file preview or statistics
    st.markdown("<h1 style='text-align: center;'>📈 Live Seizure Monitoring Simulation</h1>", unsafe_allow_html=True)
    st.markdown("---")

    df = st.session_state.uploaded_df
    X = df[significant_features]  # Features for ML model
    display_data = df[display_features]  # Features for display

    print(st.session_state.current_index, len(X))

    if st.session_state.current_index >= len(X):
        st.success("✅ Seizure Details Sent to Dashboard!")
        
        # Auto-save the file using session_id from first row
        try:
            session_id = df.iloc[0]['session_id']
            output_filename = f"{session_id}.csv"
            # Replace null or empty indicators with "Heart Rate"
            df['Indicator'] = df['Indicator'].fillna("Heart Rate")
            df['Indicator'] = df['Indicator'].replace("", "Heart Rate")
            df = df.iloc[36:].reset_index(drop=True)
            df.to_csv(output_filename, index=False)

            # ✅ Preview the saved file
            st.markdown("### 📄 Saved Output Preview")
            preview_df = pd.read_csv(output_filename)
            st.dataframe(preview_df)

            # st.success(f"✅ Results saved as: {output_filename}")
        except Exception as e:
            st.error(f"⚠️ Failed to save file: {e}")

    else:
        idx = st.session_state.current_index

        # Predict using sliding window of 3 rows if possible
        if idx >= 2:
            window = X.iloc[idx-2:idx+1].values.reshape(1, -1)
            prediction = model.predict(window)[0]
        else:
            prediction = 0  # Default prediction if not enough data

        # --- NEW: Update seizure state and get display state ---
        display_seizure = update_seizure_state(prediction)

        # Calculate and store indicator
        indicator = calculate_indicator(df, idx)
        
        # Store ACTUAL prediction and indicator in dataframe (not the display state)
        df.at[idx, "label"] = prediction  # This saves the actual model prediction
        df.at[idx, "Indicator"] = indicator
        st.session_state.uploaded_df = df
        df.to_csv(st.session_state.uploaded_path, index=False)

        # Update features history for plotting (using display features)
        # Add time in seconds (assuming 2-second intervals)
        current_time = idx * 2  # 2 seconds per data point
        
        new_row = display_data.iloc[idx].copy()
        new_row['time_seconds'] = current_time
        
        st.session_state.features_history = pd.concat([
            st.session_state.features_history,
            pd.DataFrame([new_row])
        ], ignore_index=True)

        if len(st.session_state.features_history) > 30:
            st.session_state.features_history = st.session_state.features_history.iloc[-30:]

        # UI layout
        ccol1, ccol2, ccol3, ccol4, ccol5 = st.columns([2.25, 4, 2.25, 4, 2], gap="small")
        with ccol2:
            st.subheader("🧑 Patient's Smartwatch")
        with ccol4:
            st.subheader("👩‍⚕️ Caregiver's Monitoring App")

        col1, col2, col3, col4, col5 = st.columns([2, 2.5, 2, 3, 2], gap="small")
        
        with col2:
            with st.container(height=230, border=True):
                st.write("**Hello John**")
                # Use display_seizure instead of prediction for UI
                if display_seizure:
                    st.error("⚠️ Possible seizure detected! Are you okay?")
                    btn_col1, btn_col2, btn_col3 = st.columns([3.25, 0.25, 3.5])
                    with btn_col1:
                        if st.button("I'm Okay", key=f"ok_{idx}"):
                            st.session_state.patient_status = "ok"
                    with btn_col3:
                        if st.button("Need Help", key=f"help_{idx}"):
                            st.session_state.patient_status = "help"
                else:
                    dots = "." * st.session_state.dots_count
                    st.markdown(
                        f"<div style='text-align: left; color: green; font-size: 16px;'>Monitoring in progress{dots}<br><br></div>",
                        unsafe_allow_html=True
                    )
                    st.success("✅ All vitals normal")
                    st.session_state.dots_count = (st.session_state.dots_count + 1) % 4

        with col4:
            with st.container(height=600, border=True):
                st.markdown("<h4>Patient Vitals Monitor</h4>", unsafe_allow_html=True)
                st.markdown("<div style='border-top: 1px solid #bbb; margin-top: 5px; margin-bottom: 15px;'></div>", unsafe_allow_html=True)

                # Use display_seizure instead of prediction for UI
                if display_seizure:
                    st.error(f"🚨 Alert: Seizure Detected!\n Drastic change detected in {indicator}")
                else:
                    st.info("Monitoring... All clear")

                # Display indicator if there's a significant change
                # if display_seizure:
                #     st.warning(f"⚠️ Drastic change detected in: {indicator}")

                if 'patient_status' in st.session_state:
                    if st.session_state.patient_status == "ok":
                        st.success("✅ Patient confirms they are okay.")
                    elif st.session_state.patient_status == "help":
                        st.error("🚨 Patient requested help! Immediate attention required!")

                import altair as alt

                def get_scaled_range(series, padding=5):
                    min_val = max(series.min() - padding, 0)
                    max_val = series.max() + padding
                    return (min_val, max_val)

                tcol1, tcol2, tcol3 = st.columns([3, 1.5, 3], gap="small")
                # Heart Rate Chart
                with tcol1:
                    st.write("**Heart Rate**")
                

                if 'Heart_Rate' in st.session_state.features_history.columns:
                    df_hr = st.session_state.features_history[['time_seconds', 'Heart_Rate']]
                    hr_range = get_scaled_range(df_hr['Heart_Rate'])

                    chart_hr = alt.Chart(df_hr).mark_line(color='#FF6B6B').encode(
                        x=alt.X('time_seconds:Q', 
                               title='Time (seconds)',
                               axis=alt.Axis(format='d', labelExpr='datum.value + "s"')),
                        y=alt.Y('Heart_Rate:Q', scale=alt.Scale(domain=hr_range), title=''),
                    ).properties(height=120)
                    st.altair_chart(chart_hr, use_container_width=True)
                    with tcol3:
                        # Display current heart rate value
                        current_hr = df_hr['Heart_Rate'].iloc[-1]
                        st.write(f"🫀**{current_hr:.1f} BPM**")
                    
                    
                tcol1, tcol2, tcol3 = st.columns([5, 1.5, 3.5], gap="small")

                # Respiratory Rate Chart
                with tcol1:
                    st.write("**Respiratory Rate**")
                if 'Resp_BPM' in st.session_state.features_history.columns:
                    df_resp = st.session_state.features_history[['time_seconds', 'Resp_BPM']]
                    resp_range = get_scaled_range(df_resp['Resp_BPM'])

                    chart_resp = alt.Chart(df_resp).mark_line(color='#4ECDC4').encode(
                        x=alt.X('time_seconds:Q', 
                               title='Time (seconds)',
                               axis=alt.Axis(format='d', labelExpr='datum.value + "s"')),
                        y=alt.Y('Resp_BPM:Q', scale=alt.Scale(domain=resp_range), title=''),
                    ).properties(height=120)
                    st.altair_chart(chart_resp, use_container_width=True)
                    with tcol3:
                        # Display current respiratory rate value
                        current_resp = df_resp['Resp_BPM'].iloc[-1]
                        st.write(f"🫁**{current_resp:.1f} BPM**")
                                
                #st.markdown("<div style='border-top: 1px solid #bbb; margin-top: 10px; margin-bottom: -25px;'></div>", unsafe_allow_html=True)

        # Control update speed
        if st.session_state.tval < 3:
            st.session_state.tval += 1
            time.sleep(0.0001)
        else:
            # change 1 to something less to increase speed during testing
            time.sleep(1)


        # Advance to next row and rerun
        st.session_state.current_index += 1
        st.rerun()

    # --- Reset Simulation (only show during simulation) ---
    st.divider()
    if st.button("🔄 Reset Simulation"):
        st.session_state.uploaded_df = None
        st.session_state.current_index = 0
        st.session_state.features_history = pd.DataFrame(columns=display_features + ['time_seconds'])
        st.session_state.dots_count = 0
        st.session_state.tval = 0
        st.session_state.simulation_started = False
        # Reset seizure state
        st.session_state.seizure_active = False
        st.session_state.consecutive_zeros = 0
        if 'patient_status' in st.session_state:
            del st.session_state.patient_status
        st.rerun()