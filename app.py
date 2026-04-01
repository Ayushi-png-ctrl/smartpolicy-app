import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="SmartPolicy - Insurance Premium Predictor",
    page_icon="💰",
    layout="wide"
)

tab1, tab2, tab3 = st.tabs(["Prediction", "Upload", "Sample"])

with tab1:
    st.header("Prediction")
    # ONLY upload content here

with tab2:
    st.header("Upload")
    # ONLY sample content here

with tab3:
    st.header("Sample")
    # ONLY prediction content here

# Custom CSS for better styling
with tab1:
    st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3D58;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E5A7F;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-amount {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the correct directory.")
        return None, None

# Load model
model, scaler = load_model()

# Header section
st.markdown('<h1 class="main-header">🏥 SmartPolicy</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Insurance Premium Predictor</p>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Enter Customer Details")
    
    # Input form
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        
        sex = st.selectbox("Gender", options=["male", "female"])
        
        bmi = st.number_input(
            "BMI (Body Mass Index)", 
            min_value=10.0, 
            max_value=50.0, 
            value=25.0, 
            step=0.1,
            help="BMI between 18.5-24.9 is considered healthy"
        )
        
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        
        smoker = st.selectbox("Smoker", options=["yes", "no"])
        region="northeast"
      
        
        submitted = st.form_submit_button("💰 Predict Premium", use_container_width=True)

with col2:
    st.markdown("### 📊 Quick Insights")
    
    # Display key factors affecting insurance costs
    st.info(
        """
        **💡 Did you know?**
        - Smokers pay up to 3-4x higher premiums
        - Premiums increase with age
        - Higher BMI may increase costs
        - Region has minimal impact
        """
    )
    
    # BMI Categories
    if 'bmi' in locals():
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "#FFA500"
        elif 18.5 <= bmi < 25:
            bmi_category = "Healthy Weight"
            bmi_color = "#00FF00"
        elif 25 <= bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "#FFA500"
        else:
            bmi_category = "Obese"
            bmi_color = "#FF0000"
        
        st.markdown(f"**BMI Category:** <span style='color:{bmi_color}'>{bmi_category}</span>", unsafe_allow_html=True)

if submitted and model is not None:
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'male' else 0],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'yes' else 0],
        'region':[0],
        'bmi_risk': [bmi / age]  # Feature engineering as done in your notebook
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Display prediction
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    
    with col4:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Estimated Premium</h2>
            <div class="prediction-amount">${prediction:,.2f}</div>
            <p>per year</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison with average
    if smoker == 'yes':
        avg_smoker = 32050.23
        st.metric("vs Average Smoker", f"${prediction - avg_smoker:,.2f}")
    else:
        avg_non_smoker = 8440.66
        st.metric("vs Average Non-Smoker", f"${prediction - avg_non_smoker:,.2f}")
    
    # Feature importance visualization
    st.markdown("### 🔍 Factors Influencing This Prediction")
    
    # Create a simple gauge for risk factors
    risk_factors = []
    risk_values = []
    
    if smoker == 'yes':
        risk_factors.append("Smoking")
        risk_values.append(90)
    
    if age > 50:
        risk_factors.append("Age > 50")
        risk_values.append(min((age-50)*3, 80))
    
    if bmi > 30:
        risk_factors.append("High BMI")
        risk_values.append(min((bmi-30)*5, 70))
    
    if risk_factors:
        fig = go.Figure(data=[
            go.Bar(name='Risk Level', x=risk_factors, y=risk_values, marker_color=['red' if v>70 else 'orange' for v in risk_values])
        ])
        fig.update_layout(
            title="Risk Factor Impact",
            xaxis_title="Factors",
            yaxis_title="Risk Score",
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational note
    with st.expander("📚 Understanding Your Premium"):
        st.markdown("""
        Insurance premiums are calculated based on:
        - **Age**: Older individuals typically have higher health risks
        - **Smoking**: Significantly increases health risks and costs
        - **BMI**: Higher BMI correlates with more health issues
        - **Children**: Family coverage affects premium calculations
        - **Region**: Local healthcare costs vary by location
        
        This prediction is based on historical data and machine learning analysis.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>SmartPolicy | AI-Powered Insurance Prediction | v1.0</p>",
    unsafe_allow_html=True
)
with tab2:
    import pandas as pd
    import re

    st.header("📂 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or JSON file",
        type=["csv", "json"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        st.success("✅ File uploaded successfully")
        st.dataframe(df.head())

    st.subheader("🔗 Load from Google Drive")

    drive_link = st.text_input("Paste Google Drive link")

    if drive_link:
        try:
            file_id = re.findall(r'/d/(.*?)/', drive_link)[0]
            download_url = f"https://drive.google.com/uc?id={file_id}"

            df = pd.read_csv(download_url)

            st.success("✅ File loaded from Google Drive")
            st.dataframe(df.head())

        except:
            st.error("❌ Invalid link")

with tab3:
    st.subheader("📄 Example File Format")

    example_data = pd.DataFrame({
        "age": [25, 30, 45],
        "bmi": [22.5, 27.1, 31.0],
        "smoker": ["yes", "no", "yes"],
        "charges": [2000, 3000, 5000]
    })

    st.write("This is how your uploaded file should look:")
    st.dataframe(example_data)

    # Download button
    csv = example_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇ Download Example CSV",
        data=csv,
        file_name="example_file.csv",
        mime="text/csv"
    )
            
