import os
import zipfile
import torch
import streamlit as st

# ===============================
# IMPORT EVERYTHING FROM YOUR TRAINING FILE
# ===============================
try:
    from train_bilstm_transliteration_complete import (
        BPE, Encoder, Decoder, Seq2Seq, DEVICE,
        romanize_urdu, clean_urdu
    )
    st.success("✅ Successfully imported from training file")
except ImportError as e:
    st.error(f"❌ Failed to import: {e}")
    st.stop()

# ===============================
# LOAD MODEL PROPERLY
# ===============================
@st.cache_resource
def load_model():
    try:
        # Extract model if needed
        if not os.path.exists("urdu_transliterator_best.pt") and os.path.exists("urdu_transliterator_best.zip"):
            with zipfile.ZipFile("urdu_transliterator_best.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            st.success("✅ Model extracted")

        # Load the STATE_DICT (weights only)
        state_dict = torch.load("urdu_transliterator_best.pt", map_location=DEVICE)
        
        # Recreate the model architecture first
        st.write("🔧 Recreating model architecture...")
        
        # You need to know the vocabulary sizes from your training
        # Try common sizes or detect from state_dict
        src_vocab_size = 4004  # From your earlier error
        trg_vocab_size = 4004  # From your earlier error
        
        # Recreate model with same architecture as training
        encoder = Encoder(src_vocab_size, emb_dim=128, hid_dim=256, n_layers=2, dropout=0.3)
        decoder = Decoder(trg_vocab_size, emb_dim=128, hid_dim=256, n_layers=4, dropout=0.3)
        model = Seq2Seq(encoder, decoder, enc_hid=256, dec_hid=256)
        
        # Load the weights into the model
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model
model = load_model()

# ===============================
# SIMPLE ROMANIZATION (USING YOUR EXISTING CODE)
# ===============================
def transliterate_urdu(text):
    try:
        # Use your EXACT same functions from training
        cleaned = clean_urdu(text)
        romanized = romanize_urdu(cleaned)
        return romanized
    except Exception as e:
        return f"Error: {e}"

# ===============================
# STREAMLIT UI
# ===============================
st.title("📝 Urdu → Roman Urdu Transliteration")
st.write("Using your trained model and functions")

user_input = st.text_area("Enter Urdu text:", placeholder="یہ دنیا بہت خوبصورت ہے")

if st.button("Transliterate"):
    if user_input.strip():
        output = transliterate_urdu(user_input)
        st.success("**Roman Urdu:**")
        st.markdown(f"### {output}")
        
        # Show debug info
        with st.expander("Debug Info"):
            st.write(f"Input: {user_input}")
            st.write(f"Cleaned: {clean_urdu(user_input)}")
            st.write(f"Model loaded: {model is not None}")
    else:
        st.warning("Please enter Urdu text")
