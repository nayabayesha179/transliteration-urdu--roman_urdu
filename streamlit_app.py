import os
import zipfile
import torch
import streamlit as st

# ===============================
# IMPORT EVERYTHING FROM YOUR TRAINING FILE
# ===============================
try:
    # Import all classes and functions from your training file
    from train_bilstm_transliteration_complete import (
        BPE, Encoder, Decoder, Seq2Seq, DEVICE,
        romanize_urdu, clean_urdu, 
        discover_urdu_lines, prepare_dataset_extraction,
        # Import any other functions you need
    )
    st.success("✅ Successfully imported from training file")
except ImportError as e:
    st.error(f"❌ Failed to import from training file: {e}")
    st.stop()

# ===============================
# SIMPLE STREAMLIT APP
# ===============================
@st.cache_resource
def load_model():
    try:
        # Extract model if needed
        if not os.path.exists("urdu_transliterator_best.pt") and os.path.exists("urdu_transliterator_best.zip"):
            with zipfile.ZipFile("urdu_transliterator_best.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            st.success("✅ Model extracted")

        # Load the model
        model = torch.load("urdu_transliterator_best.pt", map_location=DEVICE)
        model.eval()
        st.success("✅ Model loaded")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model
model = load_model()

if model is None:
    st.error("Failed to load model")
    st.stop()

# Simple transliteration using your existing functions
def transliterate_urdu_simple(text):
    try:
        # Use the EXACT same functions from your training
        cleaned = clean_urdu(text)
        romanized = romanize_urdu(cleaned)
        return romanized
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("📝 Urdu → Roman Urdu Transliteration")
st.write("Using functions from your training code")

user_input = st.text_area("Enter Urdu text:", placeholder="یہ دنیا بہت خوبصورت ہے")

if st.button("Transliterate"):
    if user_input.strip():
        output = transliterate_urdu_simple(user_input)
        st.success("**Roman Urdu:**")
        st.markdown(f"### {output}")
    else:
        st.warning("Please enter Urdu text")
