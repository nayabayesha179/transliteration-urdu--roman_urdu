import os
import zipfile
import torch
import streamlit as st
import pickle

# Try to import your model classes with fallbacks
try:
    from train_bilstm_transliteration_complete import Encoder, Decoder, Seq2Seq, DEVICE
except ImportError:
    st.error("❌ Could not import model classes. Please make sure all model files are available.")
    # Define minimal versions as fallback
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass
    class Decoder:
        def __init__(self, *args, **kwargs):
            pass
    class Seq2Seq:
        def __init__(self, *args, **kwargs):
            pass
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model_and_bpe():
    try:
        # === EXTRACT MODEL IF COMPRESSED ===
        model_path = "urdu_transliterator_best.pt"
        zip_path = "urdu_transliterator_best.zip"
        
        if not os.path.exists(model_path) and os.path.exists(zip_path):
            st.write("📦 Extracting model from ZIP...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")
            st.success("✅ Model extracted successfully!")
        
        # === LOAD TOKENIZERS WITH STANDARD PICKLE ===
        # Try different possible tokenizer names
        tokenizer_files = {
            "standard": ["urdu_transliterator_src_bpe_standard.pkl", "urdu_transliterator_trg_bpe_standard.pkl"],
            "original": ["urdu_transliterator_src_bpe.pkl", "urdu_transliterator_trg_bpe.pkl"],
            "fallback": ["src_bpe.pkl", "trg_bpe.pkl"]
        }
        
        src_bpe = None
        trg_bpe = None
        
        for version, (src_file, trg_file) in tokenizer_files.items():
            if os.path.exists(src_file) and os.path.exists(trg_file):
                st.write(f"📁 Loading tokenizers ({version})...")
                try:
                    with open(src_file, "rb") as f:
                        src_bpe = pickle.load(f)
                    with open(trg_file, "rb") as f:
                        trg_bpe = pickle.load(f)
                    st.success(f"✅ Tokenizers loaded successfully ({version})!")
                    break
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {version} tokenizers: {e}")
                    continue
        
        if src_bpe is None or trg_bpe is None:
            st.error("❌ Could not load any tokenizer files")
            # Show available files
            st.write("📁 Available files:")
            for file in os.listdir('.'):
                if '.pkl' in file or '.pt' in file or '.zip' in file:
                    st.write(f"   - {file}")
            return None, None, None

        # === LOAD MODEL ===
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None, None, None

        # Model architecture must match training
        model = Seq2Seq(
            Encoder(len(src_bpe.vocab), emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3),
            Decoder(len(trg_bpe.vocab), emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3),
            DEVICE
        ).to(DEVICE)

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, src_bpe, trg_bpe
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Load model and tokenizers
model, src_bpe, trg_bpe = load_model_and_bpe()

if model is None or src_bpe is None or trg_bpe is None:
    st.error("Failed to load model. Please check if all required files are uploaded.")
    st.write("**Required files:**")
    st.write("- urdu_transliterator_best.zip (compressed model)")
    st.write("- urdu_transliterator_src_bpe_standard.pkl (source tokenizer)")
    st.write("- urdu_transliterator_trg_bpe_standard.pkl (target tokenizer)")
    st.stop()

# Create reverse vocabulary
trg_rev_vocab = {v: k for k, v in trg_bpe.vocab.items()}

def transliterate_urdu(text):
    try:
        model.eval()
        text = text.strip()
        
        # Encode source text
        if hasattr(src_bpe, 'encode'):
            src_ids = src_bpe.encode(text)
        else:
            # Fallback encoding
            src_ids = [src_bpe.vocab.get(char, src_bpe.vocab.get('<unk>', 3)) for char in text.lower()]
        
        if not src_ids:
            return "No output generated"
            
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

        # Get special tokens
        trg_sos = trg_bpe.vocab.get("<sos>", 1)
        trg_eos = trg_bpe.vocab.get("<eos>", 2)
        input_tok = torch.tensor([trg_sos], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            # Encoder forward
            enc_out, (h, c) = model.encoder(src)
            
            # Handle bidirectional reduction (if method exists)
            if hasattr(model, '_reduce_bidir'):
                dec_h = model._reduce_bidir(h)
                dec_c = model._reduce_bidir(c)
            else:
                # Simple fallback
                dec_h = h
                dec_c = c

            # Adjust decoder hidden states if needed
            dec_layers = model.decoder.rnn.num_layers
            if dec_h.size(0) < dec_layers:
                last_h = dec_h[-1:].repeat(dec_layers - dec_h.size(0), 1, 1)
                dec_h = torch.cat([dec_h, last_h], dim=0)
                last_c = dec_c[-1:].repeat(dec_layers - dec_c.size(0), 1, 1)
                dec_c = torch.cat([dec_c, last_c], dim=0)

            hidden = (dec_h, dec_c)
            output_tokens = []
            
            # Decoder forward
            for _ in range(120):  # max length
                logits, hidden = model.decoder(input_tok, hidden)
                pred = logits.argmax(1)
                token = pred.item()
                
                if token == trg_eos:
                    break
                    
                output_token = trg_rev_vocab.get(token, "")
                if output_token and output_token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                    output_tokens.append(output_token)
                    
                input_tok = pred

        return " ".join(output_tokens) if output_tokens else "No output generated"
        
    except Exception as e:
        return f"Error during transliteration: {str(e)}"

# Streamlit UI
st.title("📝 Urdu → Roman Urdu Transliteration")
st.write("This app converts Urdu text into Roman Urdu using a BiLSTM seq2seq model.")

user_input = st.text_area("Enter Urdu text:", placeholder="یہ دنیا بہت خوبصورت ہے")

if st.button("Transliterate"):
    if user_input.strip():
        with st.spinner("Transliterating..."):
            output = transliterate_urdu(user_input)
        st.success("**Roman Urdu:**")
        st.markdown(f"### {output}")
    else:
        st.warning("Please enter some Urdu text first.")

# Debug info
with st.expander("Debug Info"):
    st.write("📁 Files in directory:")
    for file in os.listdir('.'):
        if '.pkl' in file or '.pt' in file or '.zip' in file:
            st.write(f" - {file}")
