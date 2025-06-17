# drug_classifier_app.py

# Importing ToolKits
import re
from time import sleep
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.components.v1 import html
import warnings

def run():
    st.set_page_config(
        page_title="Drug Classification",
        page_icon="💉🩸",
        layout="wide"
    )
    warnings.simplefilter(action='ignore', category=FutureWarning)

    @st.cache_data
    def load_model(model_path):
        return pd.read_pickle(model_path)

    model = pd.read_pickle("ML2.pkl")

    # Light and soothing color theme
    st.markdown(
        """
    <style>
         .center {
            display: flex;
            justify-content: center;
            align-items: center;    
        }
         h3 {
            font-size: 24px;
            color: #2f4f4f;
         }
         .st-emotion-cache-16txtl3 h1 {
            font: bold 28px 'Segoe UI';
            text-align: center;
            margin-bottom: 15px;
         }
         div[data-testid=stSidebarContent] {
            background-color: #f4f4f4;
            border-right: 2px solid #d3d3d3;
            padding: 10px;
         }

         .plot-container.plotly {
            border: 1px solid #ccc;
            border-radius: 6px;
         }

         div[data-baseweb=select]>div {
            cursor: pointer;
            background-color: #f0f0f0;
            border: 1px solid #d3d3d3;
        }

        div[data-baseweb=select]>div:hover {
            border: 2px solid #a0a0a0;
        }

        div[data-baseweb=base-input] {
            background-color: #fff;
            border: 1px solid #d3d3d3;
            border-radius: 5px;
            padding: 5px;
        }

        div[data-testid=stFormSubmitButton]> button {
            width: 20%;
            background: linear-gradient(to right, #a1c4fd, #c2e9fb);
            border: 1px solid #ccc;
            padding: 15px;
            border-radius: 25px;
            color: #2f4f4f;
            font-weight: bold;
        }

        div[data-testid=stFormSubmitButton]> button:hover {
            opacity: 0.9;
            border: 2px solid #2f4f4f;
            color: #2f4f4f;
        }

        div[data-testid=stFormSubmitButton]  p {
            font-weight: bold;
            font-size: 18px;
        }

        div [data-testid=stImage] {
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """,
        unsafe_allow_html=True
    )

    header = st.container(border=True)
    content = st.container(border=True)

    with header:
        st.markdown("""
        <style>
        h1 {
            color: #2f4f4f;
            background: linear-gradient(to right, #a1c4fd, #c2e9fb);
            padding: 12px;
            border-radius: 8px;
            border: 2px solid #d3d3d3;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<h1>Drug Classification 💉🩸</h1>", unsafe_allow_html=True)

    with content:
        colu0, col1, col2 = st.columns([1, 5, 1])
        with col1:
            st.markdown('<div class="center">', unsafe_allow_html=True)
            with st.form("Preidct"):
                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input('Age', min_value=1, max_value=100, value=48)
                    Cholesterol = st.selectbox('Cholesterol', options=["HIGH", "NORMAL"], index=0)
                    BP = st.selectbox('BP', options=["LOW", "NORMAL", "HIGH"], index=0)
                with c2:
                    Na_to_k = st.number_input('Na_to_k', min_value=15, max_value=75, value=25)
                    Sex = st.selectbox('Sex', options=["M", "F"], index=0)

                predict_button = st.form_submit_button("Predict")

    if predict_button:
        bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
        category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
        Age = pd.cut([age], bins=bin_age, labels=category_age)[0]

        bin_NatoK = [0, 9, 19, 29, 50]
        category_NatoK = ['<10', '10-20', '20-30', '>30']
        Na_to_k2 = pd.cut([Na_to_k], bins=bin_NatoK, labels=category_NatoK)[0]

        data = {
            'Sex': [Sex],
            'BP': [BP],
            'Cholesterol': [Cholesterol],
            'Age_binned': [Age],
            'Na_to_K_binned': [Na_to_k2]
        }
        df = pd.DataFrame(data)
        df_encoded = pd.get_dummies(df, columns=['Sex', 'BP', 'Cholesterol', 'Age_binned', 'Na_to_K_binned'])

        desired_columns_order = [
            'Sex_F', 'Sex_M',
            'BP_HIGH', 'BP_LOW', 'BP_NORMAL',
            'Cholesterol_HIGH', 'Cholesterol_NORMAL',
            'Age_binned_<20s', 'Age_binned_20s', 'Age_binned_30s', 'Age_binned_40s', 'Age_binned_50s', 'Age_binned_60s', 'Age_binned_>60s',
            'Na_to_K_binned_<10', 'Na_to_K_binned_10-20', 'Na_to_K_binned_20-30', 'Na_to_K_binned_>30'
        ]
        df_encoded = df_encoded.reindex(columns=desired_columns_order).fillna(0).astype(int)

        st.subheader("Input DataFrame:")
        styler = df_encoded.style.background_gradient(cmap='YlGnBu').set_properties(**{
            'font-size': '16px',
            'color': '#2f4f4f',
            'border': '1px solid #ccc'
        })
        st.write(styler)

        prediction = model.predict(df_encoded)
        prob = model.predict_proba(df_encoded)
        class_labels = model.classes_
        prob_df = pd.DataFrame(prob, columns=class_labels)

        st.markdown("""
        <style>
        .gradient-container {
            background: linear-gradient(to right, #a1c4fd, #c2e9fb);
            padding: 18px;
            border-radius: 15px;
            color: #2f4f4f;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            border: 2px solid #d3d3d3;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="gradient-container">{prediction[0].upper()}</div>', unsafe_allow_html=True)

        st.markdown('<p style="color:#2f4f4f;font-size:24px;margin-top:20px;"><b>Predicted Probabilities:</b></p>', unsafe_allow_html=True)
        styled_probs = prob_df.style.background_gradient(cmap='YlGnBu').set_properties(**{
            'font-size': '15px',
            'color': '#2f4f4f',
            'border': '1px solid #ccc'
        })
        st.table(styled_probs)

run()
