import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pickle, json
from huggingface_hub import hf_hub_download

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Neural Storyteller",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------

st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #4F8BF9;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9aa0a6;
    margin-bottom: 40px;
}

.card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #30363d;
}

.caption-box {
    background: linear-gradient(135deg,#4F8BF9,#7B61FF);
    padding: 25px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 500;
    color: white;
}

.generate-btn button{
    width: 100%;
    height: 55px;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Vocabulary class (needed for pickle)
# -------------------------------

class Vocabulary:
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)

# -------------------------------
# Model Architecture
# -------------------------------

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512, dropout=0.3):
        super().__init__()

        self.fc = nn.Linear(feature_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(torch.relu(self.bn(self.fc(x))))


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512,
                 num_layers=2, dropout=0.3, pad_idx=0):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_hidden(self, enc_out):

        h0 = enc_out.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        return h0, c0


class NeuralStoryteller(nn.Module):

    def __init__(self, vocab_size,
                 embed_dim=256,
                 hidden_size=512,
                 num_layers=2,
                 dropout=0.3):

        super().__init__()

        self.encoder = ImageEncoder(2048, hidden_size, dropout)

        self.decoder = CaptionDecoder(
            vocab_size,
            embed_dim,
            hidden_size,
            num_layers,
            dropout
        )

# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():

    REPO_ID = "Usman366/neural-storyteller"

    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.pkl")
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="model_weights.pt")

    with open(config_path) as f:
        cfg = json.load(f)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    model = NeuralStoryteller(vocab_size=len(vocab))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    return model, vocab

model, vocab = load_model()

# -------------------------------
# Feature extractor
# -------------------------------

backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
extractor = nn.Sequential(*list(backbone.children())[:-1])
extractor.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

def extract_features(img):

    img = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        feat = extractor(img).view(1,-1)

    return feat


# -------------------------------
# Caption generation
# -------------------------------

def generate_caption(image):

    feat = extract_features(image)

    enc = model.encoder(feat)
    h, c = model.decoder.init_hidden(enc)

    word = vocab.stoi['<start>']

    result = []

    for _ in range(30):

        inp = torch.tensor([[word]])
        emb = model.decoder.embed(inp)

        out, (h, c) = model.decoder.lstm(emb, (h, c))

        word = model.decoder.fc_out(out.squeeze(1)).argmax(-1).item()

        if word == vocab.stoi['<end>']:
            break

        if word in vocab.itos:
            result.append(vocab.itos[word])

    # ---- Prevent empty caption ----
    if len(result) == 0:
        return "Model could not generate a caption."

    return " ".join(result)


# -------------------------------
# UI Layout
# -------------------------------

# -----------------------------
# HEADER
# -----------------------------

st.markdown('<p class="title">Neural Storyteller</p>', unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">AI Image Captioning using ResNet50 + LSTM</p>',
unsafe_allow_html=True
)

# -----------------------------
# MAIN LAYOUT
# -----------------------------

left, right = st.columns([1.2,1])

# -----------------------------
# IMAGE UPLOAD PANEL
# -----------------------------

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Upload Image")

    uploaded = st.file_uploader(
        "Drag & Drop or Browse Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        st.image(image, width="stretch")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# CAPTION GENERATION PANEL
# -----------------------------

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Generate Caption")

    if uploaded:

        if st.button("Generate Caption", key="generate"):

            with st.spinner("AI is analyzing the image..."):

                caption = generate_caption(image)

            st.markdown("### Generated Caption")

            st.markdown(
                f'<div class="caption-box">{caption}</div>',
                unsafe_allow_html=True
            )

    else:
        st.info("Upload an image first.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray'>
Built with ❤️ using Streamlit | Neural Storyteller
</div>
""",
unsafe_allow_html=True
)