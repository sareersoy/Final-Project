import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoModel, AutoTokenizer
import torch
import pandas as pd

# Load the models
# Set page configuration
st.set_page_config(page_title="Text Classification App", page_icon=":sparkles:", layout="wide")
# Custom CSS for styling

model_path_eng = r"C:\Users\saren\Desktop\modeldeneme\distilbert_model_eng"
model_path_spa = r"C:\Users\saren\Desktop\modeldeneme\distilbert_model_es"

tokenizer_en = DistilBertTokenizer.from_pretrained(model_path_eng)
model_en = DistilBertForSequenceClassification.from_pretrained(model_path_eng)

tokenizer_es = DistilBertTokenizer.from_pretrained(model_path_spa)
model_es = DistilBertForSequenceClassification.from_pretrained(model_path_spa)

st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        color: #FF6347;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .expander-header {
        font-size: 20px;
        color: #FF6347;
    }
   .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 5px;
            text-decoration: none;
        }
    .button:hover {
        background-color: #45a049;
    }
 
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='main-title'>Text Classification App</div>", unsafe_allow_html=True)
st.markdown("""
    <div class='description'>
        Welcome to the Text Classification App! 
        This tool allows you to classify text data as 'Real Name' or 'Random Word'.<br>
        You can either enter text manually or upload a CSV file for batch processing.
    </div>
""", unsafe_allow_html=True)

# Option to choose language
st.markdown("<div class='section-header'>Step 1: Select Language</div>", unsafe_allow_html=True)
lang = st.radio("", ('English', 'Spanish'))

# Setup tokenizer and model based on language selection
tokenizer = tokenizer_en if lang == 'English' else tokenizer_es
model = model_en if lang == 'English' else model_es

# User input for text
st.markdown("<div class='section-header'>Step 2: Enter Text for Classification</div>", unsafe_allow_html=True)
user_input = st.text_input("Enter text to classify:")

if st.button('Classify Text', key='classify_text_button'):
    if user_input:
        # Encode the text using the appropriate tokenizer
        inputs = tokenizer(user_input, return_tensors="pt")
        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        # Assuming class 0 is 'gibberish' and class 1 is 'real name'
        is_real_name = probs[:, 1] > 0.5
        result = "Real Name" if is_real_name else "Random Word"
        st.success(f"Classification: {result}")
    else:
        st.warning("Please enter some text to classify.")

st.markdown("<div class='section-header'>Step 3: Upload CSV File for Batch Classification</div>",
            unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    delimiter = st.radio("Select the delimiter used in the CSV file", (',', ';', '\t', '|'))
    data = pd.read_csv(uploaded_file, delimiter=delimiter)

    st.markdown("<div class='section-header'>Step 4: Select Column to Classify</div>", unsafe_allow_html=True)
    # Display the column names and let the user select which column to use
    column_name = st.selectbox("Select the column to classify", data.columns)

    if column_name:
        outputs = []
        for text in data[column_name]:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            is_real_name = probs[:, 1] > 0.5
            outputs.append("Real Name" if is_real_name else "Random Word")

        # Create a new DataFrame with only the selected column and classification results
        result_data = pd.DataFrame({column_name: data[column_name], 'classification': outputs})

        # Display results in an expandable section
        with st.expander("See Classification Results", expanded=True):
            st.markdown("<div class='expander-header'>Classification Results</div>", unsafe_allow_html=True)
            st.dataframe(result_data.style.set_table_styles(
                [{'selector': 'thead th', 'props': [('background-color', '#FF6347'), ('color', 'white')]}]
            ))

        st.download_button(
            label="Download Results as CSV",
            data=result_data.to_csv(index=False).encode('utf-8'),
            file_name='classified.csv',
            mime='text/csv',
            key='download_csv_button'
        )
    else:
        st.warning("Please select a column to classify.")
