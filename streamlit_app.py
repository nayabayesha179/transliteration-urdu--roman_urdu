import os
import zipfile
import torch
import streamlit as st
import pickle

# Import from your training file
from train_bilstm_transliteration_complete import (
    Encoder, Decoder, Seq2Seq, DEVICE, 
    BPE, romanize_urdu, clean_urdu
)

@st.cache_resource
def load_model_and_tokenizers():
    try:
        # Extract model if needed
        if not os.path.exists("urdu_transliterator_best.pt") and os.path.exists("urdu_transliterator_best.zip"):
            with zipfile.ZipFile("urdu_transliterator_best.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            st.success("✅ Model extracted from ZIP")

        # Load the EXACT tokenizers from training
        st.write("📥 Loading original training tokenizers...")
        
        # Try different possible tokenizer files
        tokenizer_files = [
            ("src_bpe.pkl", "trg_bpe.pkl"),
            ("urdu_transliterator_src_bpe.pkl", "urdu_transliterator_trg_bpe.pkl"),
        ]
        
        src_bpe = None
        trg_bpe = None
        
        for src_file, trg_file in tokenizer_files:
            if os.path.exists(src_file) and os.path.exists(trg_file):
                try:
                    with open(src_file, "rb") as f:
                        src_bpe = pickle.load(f)
                    with open(trg_file, "rb") as f:
                        trg_bpe = pickle.load(f)
                    st.success(f"✅ Loaded {src_file}, {trg_file}")
                    st.write(f"Source vocab: {len(src_bpe.vocab)}, Target vocab: {len(trg_bpe.vocab)}")
                    break
                except Exception as e:
                    st.warning(f"Failed to load {src_file}: {e}")
                    continue
        
        if src_bpe is None or trg_bpe is None:
            st.error("❌ Could not load tokenizers. Please upload src_bpe.pkl and trg_bpe.pkl")
            return None, None, None

        # Load model with EXACT architecture
        st.write("🧠 Loading model...")
        model = Seq2Seq(
            Encoder(len(src_bpe.vocab), emb_dim=128, hid_dim=256, n_layers=2, dropout=0.3),
            Decoder(len(trg_bpe.vocab), emb_dim=128, hid_dim=256, n_layers=4, dropout=0.3),
            enc_hid=256, dec_hid=256
        ).to(DEVICE)

        model.load_state_dict(torch.load("urdu_transliterator_best.pt", map_location=DEVICE))
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, src_bpe, trg_bpe
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None, None

# Load model
model, src_bpe, trg_bpe = load_model_and_tokenizers()

if model is None:
    st.error("Failed to load model")
    st.stop()

# Create reverse vocabulary
trg_rev_vocab = {v: k for k, v in trg_bpe.vocab.items()}

def transliterate_urdu(text):
    try:
        model.eval()
        text = text.strip()
        
        # Use the SAME preprocessing as training
        cleaned_text = clean_urdu(text)
        roman_text = romanize_urdu(cleaned_text).lower()
        
        st.write(f"🔍 Preprocessed: {roman_text}")
        
        # Encode using the EXACT same BPE tokenizer
        src_ids = src_bpe.encode(roman_text)
        st.write(f"🔍 Encoded IDs: {src_ids}")
        
        if not src_ids:
            return "No output generated"
            
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

        trg_sos = trg_bpe.vocab["<sos>"]
        trg_eos = trg_bpe.vocab["<eos>"]
        input_tok = torch.tensor([trg_sos], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            enc_out, (h, c) = model.encoder(src)
            
            # Handle bidirectional reduction
            if hasattr(model, '_reduce'):
                dec_h = model._reduce(h)
                dec_c = model._reduce(c)
            else:
                # Manual reduction
                n2, B, H = h.size()
                n = n2 // 2
                h_combined = h.view(n, 2, B, H)
                dec_h = torch.cat([h_combined[:,0,:,:], h_combined[:,1,:,:]], dim=2)
                c_combined = c.view(n, 2, B, H)
                dec_c = torch.cat([c_combined[:,0,:,:], c_combined[:,1,:,:]], dim=2)

            # Adjust hidden states
            dec_layers = model.decoder.rnn.num_layers
            if dec_h.size(0) < dec_layers:
                last_h = dec_h[-1:].repeat(dec_layers - dec_h.size(0), 1, 1)
                dec_h = torch.cat([dec_h, last_h], dim=0)
                last_c = dec_c[-1:].repeat(dec_layers - dec_c.size(0), 1, 1)
                dec_c = torch.cat([dec_c, last_c], dim=0)

            hidden = (dec_h, dec_c)
            output_tokens = []
            
            for _ in range(120):
                logits, hidden = model.decoder(input_tok, hidden)
                pred = logits.argmax(1)
                token = pred.item()
                if token == trg_eos:
                    break
                output_token = trg_rev_vocab.get(token, "<unk>")
                output_tokens.append(output_token)
                input_tok = pred

        result = " ".join(output_tokens)
        st.write(f"🔍 Raw output tokens: {output_tokens}")
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("📝 Urdu → Roman Urdu Transliteration")
st.write("Using your trained BiLSTM model")

user_input = st.text_area("Enter Urdu text:", placeholder="یہ دنیا بہت خوبصورت ہے")

if st.button("Transliterate"):
    if user_input.strip():
        with st.spinner("Processing..."):
            output = transliterate_urdu(user_input)
        st.success("**Roman Urdu:**")
        st.markdown(f"### {output}")
    else:
        st.warning("Please enter Urdu text")
