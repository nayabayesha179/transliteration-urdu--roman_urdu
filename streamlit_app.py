import streamlit as st
import torch
from train_bilstm_transliteration_complete import Encoder, Decoder, Seq2Seq, BPE, DEVICE

# Load trained model and BPE vocabularies
# (same code I gave you earlier)

st.title("📝 Urdu → Roman Urdu Transliteration")
st.write("This app converts Urdu text into its Roman Urdu form using a BiLSTM sequence-to-sequence model.")

user_input = st.text_area("Type Urdu text here:", placeholder="یہ دنیا بہت خوبصورت ہے")

if st.button("Transliterate"):
    if user_input.strip():
        with st.spinner("Processing..."):
            output = transliterate_urdu(user_input)
        st.success("Roman Urdu Transliteration:")
        st.markdown(f"**{output}**")
    else:
        st.warning("Please enter Urdu text before clicking Transliterate.")
