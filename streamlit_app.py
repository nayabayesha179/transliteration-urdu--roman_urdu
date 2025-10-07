import os
import zipfile
import torch
import streamlit as st
import pickle
import re
from collections import Counter
from typing import List

# ===============================
# BPE CLASS DEFINITION (INCLUDED IN APP)
# ===============================
class BPE:
    def __init__(self, target_vocab_size=8000, min_freq=2):
        self.target_vocab_size = target_vocab_size
        self.min_freq = min_freq
        self.merges = []
        self.vocab = {}
        self.rev = {}
    
    def __getstate__(self):
        return {
            'target_vocab_size': self.target_vocab_size,
            'min_freq': self.min_freq,
            'merges': self.merges,
            'vocab': self.vocab,
            'rev': self.rev
        }
    
    def __setstate__(self, state):
        self.target_vocab_size = state['target_vocab_size']
        self.min_freq = state['min_freq']
        self.merges = state['merges']
        self.vocab = state['vocab']
        self.rev = state['rev']
    
    def learn(self, texts):
        # Build word-frequency with character sequences
        word_freq = Counter()
        for txt in texts:
            for w in txt.strip().split():
                if not w:
                    continue
                token = ' '.join(list(w)) + ' </w>'
                word_freq[token] += 1
        merges = []
        corpus = word_freq
        while True:
            pairs = Counter()
            for word, freq in corpus.items():
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            if not pairs:
                break
            (a,b), freq = pairs.most_common(1)[0]
            if freq < self.min_freq:
                break
            merges.append((a,b))
            pat = re.escape(a + ' ' + b)
            new_corpus = {}
            for w,freqw in corpus.items():
                new_corpus[re.sub(pat, a+b, w)] = freqw
            corpus = new_corpus
            if len(merges) >= self.target_vocab_size:
                break
        self.merges = merges
        # build token list
        token_counter = Counter()
        for word, freq in corpus.items():
            for t in word.split():
                token_counter[t] += freq
        specials = ['<pad>','<sos>','<eos>','<unk>']
        tokens = specials + [t for t,_ in token_counter.most_common(self.target_vocab_size)]
        self.vocab = {t:i for i,t in enumerate(tokens)}
        self.rev = {i:t for t,i in self.vocab.items()}
        return self.vocab

    def encode_word(self, word: str) -> List[str]:
        symbols = list(word) + ['</w>']
        # apply merges greedily in learned order
        made_change = True
        while True:
            made = False
            for a,b in self.merges:
                i = 0
                while i < len(symbols)-1:
                    if symbols[i]==a and symbols[i+1]==b:
                        symbols[i] = a+b
                        del symbols[i+1]
                        made = True
                    else:
                        i += 1
            if not made:
                break
        if symbols and symbols[-1]=='</w>':
            symbols = symbols[:-1]
        return symbols

    def encode(self, text: str) -> List[int]:
        out=[]
        for w in text.strip().split():
            toks = self.encode_word(w)
            if not toks:
                out.append(self.vocab.get('<unk>'))
            else:
                for t in toks:
                    out.append(self.vocab.get(t, self.vocab.get('<unk>')))
        return out

# ===============================
# MODEL IMPORTS WITH FALLBACKS
# ===============================
try:
    from train_bilstm_transliteration_complete import Encoder, Decoder, Seq2Seq, DEVICE
except ImportError:
    st.warning("⚠️ Could not import model classes. Using fallback definitions.")
    
    # Fallback model definitions
    import torch.nn as nn
    
    class Encoder(nn.Module):
        def __init__(self, input_dim, emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
            self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, 
                              dropout=dropout, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, src):
            emb = self.dropout(self.embedding(src))
            outputs, (h, c) = self.rnn(emb)
            return outputs, (h, c)
    
    class Decoder(nn.Module):
        def __init__(self, output_dim, emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
            self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, 
                              dropout=dropout, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_dim)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, token_in, hidden):
            emb = self.dropout(self.embedding(token_in).unsqueeze(1))
            out, hidden = self.rnn(emb, hidden)
            logits = self.fc(out.squeeze(1))
            return logits, hidden
    
    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
            
        def _reduce_bidir(self, h):
            # Simple bidirectional reduction
            n2, B, H = h.size()
            n = n2 // 2
            h = h.view(n, 2, B, H)
            h_combined = torch.cat([h[:,0,:,:], h[:,1,:,:]], dim=2)
            return h_combined
            
        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            B = src.size(0)
            T = trg.size(1)
            V = self.decoder.fc.out_features
            outputs = torch.zeros(B, T, V, device=src.device)
            
            enc_out, (h, c) = self.encoder(src)
            dec_h = self._reduce_bidir(h)
            dec_c = self._reduce_bidir(c)
            
            dec_need = self.decoder.rnn.num_layers
            if dec_h.size(0) < dec_need:
                last_h = dec_h[-1:].repeat(dec_need - dec_h.size(0), 1, 1)
                dec_h = torch.cat([dec_h, last_h], dim=0)
                last_c = dec_c[-1:].repeat(dec_need - dec_c.size(0), 1, 1)
                dec_c = torch.cat([dec_c, last_c], dim=0)
                
            hidden = (dec_h.contiguous(), dec_c.contiguous())
            input_tok = trg[:,0]
            
            for t in range(1, T):
                logits, hidden = self.decoder(input_tok, hidden)
                outputs[:,t] = logits
                top1 = logits.argmax(1)
                teacher_force = (torch.rand(1).item() < teacher_forcing_ratio)
                input_tok = trg[:,t] if teacher_force else top1
                
            return outputs
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
# MODEL LOADING FUNCTION
# ===============================
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
        st.write("📁 Loading tokenizers...")
        
        # Try to load the standard tokenizers first
        src_bpe_path = "urdu_transliterator_src_bpe_standard.pkl"
        trg_bpe_path = "urdu_transliterator_trg_bpe_standard.pkl"
        
        if os.path.exists(src_bpe_path) and os.path.exists(trg_bpe_path):
            with open(src_bpe_path, "rb") as f:
                src_bpe = pickle.load(f)
            with open(trg_bpe_path, "rb") as f:
                trg_bpe = pickle.load(f)
            st.success("✅ Tokenizers loaded successfully!")
        else:
            st.error("❌ Tokenizer files not found")
            return None, None, None

        # === LOAD MODEL ===
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None, None, None

        # Create model with correct architecture
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

# ===============================
# LOAD MODEL AND SETUP
# ===============================
model, src_bpe, trg_bpe = load_model_and_bpe()

if model is None or src_bpe is None or trg_bpe is None:
    st.error("Failed to load model. Please check if all required files are uploaded.")
    st.write("**Required files:**")
    st.write("- urdu_transliterator_best.zip (compressed model)")
    st.write("- urdu_transliterator_src_bpe_standard.pkl (source tokenizer)")
    st.write("- urdu_transliterator_trg_bpe_standard.pkl (target tokenizer)")
    
    # Show available files for debugging
    st.write("📁 Available files:")
    for file in os.listdir('.'):
        if any(ext in file for ext in ['.pkl', '.pt', '.zip']):
            st.write(f"   - {file}")
    st.stop()

# Create reverse vocabulary
trg_rev_vocab = {v: k for k, v in trg_bpe.vocab.items()}

# ===============================
# TRANSLITERATION FUNCTION
# ===============================
def transliterate_urdu(text):
    try:
        model.eval()
        text = text.strip()
        
        # Encode source text
        src_ids = src_bpe.encode(text)
        
        if not src_ids:
            return "No output generated"
            
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

        # Get special tokens
        trg_sos = trg_bpe.vocab["<sos>"]
        trg_eos = trg_bpe.vocab["<eos>"]
        input_tok = torch.tensor([trg_sos], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            # Encoder forward
            enc_out, (h, c) = model.encoder(src)
            
            # Handle bidirectional reduction
            dec_h = model._reduce_bidir(h)
            dec_c = model._reduce_bidir(c)

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

# ===============================
# STREAMLIT UI
# ===============================
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
