import os
import zipfile
import torch
import streamlit as st
from train_bilstm_transliteration_complete import Encoder, Decoder, Seq2Seq, DEVICE

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
        
        # === LOAD TOKENIZERS ===
        import pickle
        with open("urdu_transliterator_src_bpe.pkl", "rb") as f:
            src_bpe = pickle.load(f)
        with open("urdu_transliterator_trg_bpe.pkl", "rb") as f:
            trg_bpe = pickle.load(f)

        st.write("✅ Tokenizers loaded successfully!")
        st.write(f"Source vocab size: {len(src_bpe.vocab)}")
        st.write(f"Target vocab size: {len(trg_bpe.vocab)}")

        # === LOAD MODEL ===
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            # Show available files
            st.write("📁 Available files:")
            for file in os.listdir('.'):
                st.write(f"   - {file}")
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
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

# Load model and tokenizers
model, src_bpe, trg_bpe = load_model_and_bpe()

if model is None or src_bpe is None or trg_bpe is None:
    st.error("Failed to load model. Please check if all required files are uploaded.")
    st.write("**Required files:**")
    st.write("- urdu_transliterator_best.zip (compressed model)")
    st.write("- urdu_transliterator_src_bpe.pkl (source tokenizer)")
    st.write("- urdu_transliterator_trg_bpe.pkl (target tokenizer)")
    st.stop()

trg_rev_vocab = {v: k for k, v in trg_bpe.vocab.items()}

def transliterate_urdu(text):
    model.eval()
    text = text.strip()
    src_ids = src_bpe.encode(text)
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    trg_sos = trg_bpe.vocab["<sos>"]
    trg_eos = trg_bpe.vocab["<eos>"]
    input_tok = torch.tensor([trg_sos], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        enc_out, (h, c) = model.encoder(src)
        dec_h = model._reduce_bidir(h)
        dec_c = model._reduce_bidir(c)

        # adjust decoder hidden states if fewer than decoder layers
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
            output_tokens.append(trg_rev_vocab.get(token, ""))
            input_tok = pred

    return " ".join(output_tokens)

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

# Debug info (optional - remove in production)
with st.expander("Debug Info"):
    st.write("📁 Files in directory:")
    for file in os.listdir('.'):
        st.write(f" - {file}")
