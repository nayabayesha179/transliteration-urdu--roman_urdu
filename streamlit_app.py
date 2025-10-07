import os
import zipfile
import torch
import streamlit as st
from train_bilstm_transliteration_complete import Encoder, Decoder, Seq2Seq, DEVICE


@st.cache_resource
def load_model_and_bpe():

        # === UNZIP THE MODEL IF NOT ALREADY ===
    if not os.path.exists("exp_small_best.pt") and os.path.exists("exp_small_best.zip"):
        with zipfile.ZipFile("exp_small_best.zip", "r") as zip_ref:
            zip_ref.extractall(".")
            st.write("✅ Model extracted from ZIP")
   
    import pickle

    with open("src_bpe.pkl", "rb") as f:
            src_bpe = pickle.load(f)
    with open("trg_bpe.pkl", "rb") as f:
            trg_bpe = pickle.load(f)


    # Model must match training architecture
    model = Seq2Seq(
        Encoder(len(src_bpe.vocab), emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3),
        Decoder(len(trg_bpe.vocab), emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3),
        DEVICE
    ).to(DEVICE)

    model.load_state_dict(torch.load("exp_small_best.pt", map_location=DEVICE))
    model.eval()
    return model, src_bpe, trg_bpe

model, src_bpe, trg_bpe = load_model_and_bpe()
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

