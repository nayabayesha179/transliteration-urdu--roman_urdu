import os
import zipfile
import torch
import streamlit as st

# Import from your training file
from train_bilstm_transliteration_complete import (
    Encoder, Decoder, Seq2Seq, DEVICE, 
    BPE, romanize_urdu  # Import BPE class and romanizer from your training file
)

@st.cache_resource
def load_model_and_tokenizers():
    try:
        # Extract model if needed
        if not os.path.exists("urdu_transliterator_best.pt") and os.path.exists("urdu_transliterator_best.zip"):
            with zipfile.ZipFile("urdu_transliterator_best.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            st.success("✅ Model extracted from ZIP")

        # Create tokenizers with correct vocabulary size (4000)
        st.write("🔄 Creating tokenizers...")
        
        src_bpe = BPE(target_vocab_size=4000, min_freq=2)
        trg_bpe = BPE(target_vocab_size=4000, min_freq=2)
        
        # Create basic vocabulary that matches your training
        base_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        
        # Add common characters
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789 .,!?-'
        for i, char in enumerate(chars):
            base_vocab[char] = i + 4
        
        src_bpe.vocab = base_vocab.copy()
        src_bpe.rev = {v: k for k, v in base_vocab.items()}
        
        trg_bpe.vocab = base_vocab.copy()
        trg_bpe.rev = {v: k for k, v in base_vocab.items()}
        
        st.success("✅ Tokenizers created")

        # Load model with correct architecture
        st.write("🧠 Loading model...")
        model = Seq2Seq(
            Encoder(len(src_bpe.vocab), emb_dim=128, hid_dim=256, n_layers=2, dropout=0.3),
            Decoder(len(trg_bpe.vocab), emb_dim=128, hid_dim=256, n_layers=2, dropout=0.3),
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
        
        # Use your existing romanize_urdu function for preprocessing
        roman_text = romanize_urdu(text)
        
        # Encode using BPE
        src_ids = src_bpe.encode(roman_text.lower())
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

        trg_sos = trg_bpe.vocab["<sos>"]
        trg_eos = trg_bpe.vocab["<eos>"]
        input_tok = torch.tensor([trg_sos], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            enc_out, (h, c) = model.encoder(src)
            dec_h = model._reduce_bidir(h)
            dec_c = model._reduce_bidir(c)

            # Adjust hidden states if needed
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
