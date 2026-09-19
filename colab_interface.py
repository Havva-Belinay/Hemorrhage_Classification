"""
colab_interface.py
Hemorrhage Detection — Gradio Arayüzü

Colab'da çalıştırma:
    !python /content/colab_interface.py
"""

import os
os.system("pip install -q gradio")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny
import gradio as gr

# ── Yollar ────────────────────────────────────────────────────────────────
if os.path.exists("/content/drive/MyDrive/hemorrhage_detection_results"):
    SAVE_DIR = "/content/drive/MyDrive/hemorrhage_detection_results"
else:
    SAVE_DIR = "/content/hemorrhage_detection_results"

CUSTOM_CKPT = os.path.join(SAVE_DIR, "custom_cnn_best.pth")
CONVNEXT_CKPT = os.path.join(SAVE_DIR, "convnext_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Hemorrhage"]
print(f"Device: {DEVICE}  |  SAVE_DIR: {SAVE_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# Model tanımları
# ══════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x): return self.block(x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32), ConvBlock(32, 64),
            ConvBlock(64, 128), ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

def build_convnext(num_classes=2, dropout_rate=0.5):
    model = convnext_tiny(weights=None)
    in_f = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.LayerNorm(in_f), nn.Dropout(dropout_rate),
        nn.Linear(in_f, 256), nn.GELU(), nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes),
    )
    return model

# ══════════════════════════════════════════════════════════════════════════
# Model yükleme
# ══════════════════════════════════════════════════════════════════════════
def load_model(model_obj, ckpt_path, name):
    if not os.path.exists(ckpt_path):
        print(f"  [!] {name} bulunamadı: {ckpt_path}")
        return None
    model_obj.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model_obj.eval().to(DEVICE)
    print(f"  [+] {name} yüklendi")
    return model_obj

print("Modeller yükleniyor...")
custom_model = load_model(CustomCNN(), CUSTOM_CKPT, "Custom CNN")
convnext_model = load_model(build_convnext(), CONVNEXT_CKPT, "ConvNeXt-Tiny")

MODELS = {}
if custom_model:   MODELS["Custom CNN"] = custom_model
if convnext_model: MODELS["ConvNeXt-Tiny"] = convnext_model
if not MODELS:
    raise RuntimeError("Model bulunamadı! Önce colab_train.py çalıştır.")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════════════════════
# Tahmin
# ══════════════════════════════════════════════════════════════════════════
def predict(image, model_name):
    if image is None:
        return placeholder_html()

    model = MODELS[model_name]
    img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    img = img.convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    p_normal = probs[0] * 100
    p_hem = probs[1] * 100

    if pred_idx == 1:
        accent = "#D94F4F"
        accent_light = "#FEE8E8"
        tag = "HEMORRHAGE"
        desc = "Kanama bulgusu tespit edildi"
        ring_grad = f"conic-gradient(#D94F4F 0% {p_hem}%, #E0F2EE {p_hem}% 100%)"
    else:
        accent = "#1D9E85"
        accent_light = "#D5F5ED"
        tag = "NORMAL"
        desc = "Kanama bulgusu tespit edilmedi"
        ring_grad = f"conic-gradient(#1D9E85 0% {p_normal}%, #F0F0F0 {p_normal}% 100%)"

    return f"""
    <div style="
        font-family:'DM Sans',sans-serif;
        display:flex; flex-direction:row; align-items:center;
        padding:20px; gap:22px; height:100%;
    ">
        <!-- Ring -->
        <div style="flex-shrink:0;">
            <div style="
                width:120px; height:120px; border-radius:50%;
                background:{ring_grad};
                display:flex; align-items:center; justify-content:center;
                box-shadow: 0 0 0 4px {accent}18;
            ">
                <div style="
                    width:96px; height:96px; border-radius:50%;
                    background:#fff;
                    display:flex; align-items:center; justify-content:center;
                ">
                    <span style="font-size:30px; font-weight:800; color:{accent};">
                        %{conf*100:.0f}
                    </span>
                </div>
            </div>
        </div>

        <!-- Info -->
        <div style="flex:1;">
            <span style="
                display:inline-block;
                padding:3px 14px; border-radius:14px;
                font-size:11px; font-weight:800; letter-spacing:1.5px;
                color:{accent}; background:{accent_light};
                margin-bottom:8px;
            ">{tag}</span>

            <div style="font-size:14px; font-weight:600; color:#374151; margin-bottom:2px;">
                {desc}
            </div>
            <div style="font-size:11px; color:#9CA3AF; margin-bottom:14px;">
                Model: {model_name}
            </div>

            <!-- Bars -->
            <div>
                <div style="display:flex; justify-content:space-between;
                            font-size:14px; margin-bottom:4px;">
                    <span style="font-weight:700; color:#1D9E85;">Normal</span>
                    <span style="font-weight:800; color:#1D9E85;">{p_normal:.1f}%</span>
                </div>
                <div style="height:14px; background:#D0E5DF; border-radius:7px;
                            overflow:hidden; margin-bottom:12px;">
                    <div style="height:100%; width:{p_normal:.1f}%;
                                background:linear-gradient(90deg,#1D9E85,#3CC8A9);
                                border-radius:7px;"></div>
                </div>
                <div style="display:flex; justify-content:space-between;
                            font-size:14px; margin-bottom:4px;">
                    <span style="font-weight:700; color:#D94F4F;">Hemorrhage</span>
                    <span style="font-weight:800; color:#D94F4F;">{p_hem:.1f}%</span>
                </div>
                <div style="height:14px; background:#F5D5D5; border-radius:7px;
                            overflow:hidden;">
                    <div style="height:100%; width:{p_hem:.1f}%;
                                background:linear-gradient(90deg,#D94F4F,#F08080);
                                border-radius:7px;"></div>
                </div>
            </div>
        </div>
    </div>
    """


def placeholder_html():
    return """
    <div style="
        display:flex; align-items:center; justify-content:center;
        height:100%; min-height:260px;
        color:#9CB5AE; font-family:'DM Sans',sans-serif; gap:8px;
    ">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
             stroke="#9CB5AE" stroke-width="1.5" stroke-linecap="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        <span style="font-size:13px;">Görüntü yükleyin</span>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════
# CSS — force side-by-side even in Colab iframe
# ══════════════════════════════════════════════════════════════════════════
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Outfit:wght@700&display=swap');

/* Container */
.gradio-container {
    max-width: 1060px !important;
    margin: auto !important;
    font-family: 'DM Sans', sans-serif !important;
    background: #F4F9F7 !important;
}

/* Force rows to never collapse */
.gradio-container .contain > .gap > div.row {
    flex-wrap: nowrap !important;
}

/* Header */
#hdr {
    background: linear-gradient(135deg, #14453D, #1D7A6A);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 4px;
    display: flex; align-items: center; gap: 12px;
}
#hdr .ico {
    width:34px; height:34px; border-radius:8px;
    background:rgba(255,255,255,0.12);
    display:flex; align-items:center; justify-content:center;
    font-size:17px;
}
#hdr h1 {
    margin:0; font-family:'Outfit',sans-serif;
    font-size:16px; font-weight:700; color:#F0FDFA;
}
#hdr p { margin:1px 0 0; font-size:11px; color:#86D4C1; }

/* Compact image upload */
#img-area { max-height: 260px; }
#img-area img { max-height: 220px !important; object-fit: contain; }

/* Button */
#go {
    background:#1D7A6A !important; border:none !important;
    border-radius:8px !important; font-weight:700 !important;
    font-size:13px !important; padding:8px 0 !important;
    color:#fff !important; width:100% !important;
}
#go:hover { background:#14614F !important; }

/* Result card */
#res {
    background:#fff; border:1px solid #D0E5DF;
    border-radius:10px;
}

/* Footer */
#ft {
    text-align:center; padding:6px 0 0;
    font-size:10px; color:#B5C4BF;
}

/* Compact spacing overrides */
.gradio-container .gap { gap: 4px !important; }
"""

# ══════════════════════════════════════════════════════════════════════════
# Arayüz
# ══════════════════════════════════════════════════════════════════════════
model_choices = list(MODELS.keys())

with gr.Blocks(
    title="Head CT Hemorrhage Detection",
    css=custom_css,
    theme=gr.themes.Soft(
        font=["DM Sans", "system-ui", "sans-serif"],
        primary_hue="teal",
        neutral_hue="gray",
        radius_size="md",
    ),
) as demo:

    gr.HTML("""
        <div id="hdr">
            <div class="ico">🧠</div>
            <div>
                <h1>Head CT Hemorrhage Detection</h1>
                <p>Beyin BT görüntülerinde kanama tespiti</p>
            </div>
        </div>
    """)

    # Ana satır — sol: görüntü+kontrol, sağ: sonuç
    with gr.Row(equal_height=True):

        # SOL PANEL
        with gr.Column(scale=1, min_width=400):
            image_input = gr.Image(
                type="pil", label="BT Görüntüsü",
                height=240, sources=["upload", "clipboard"],
                elem_id="img-area",
            )
            # Model + buton aynı satırda
            with gr.Row():
                model_radio = gr.Radio(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Model", scale=3,
                )
                predict_btn = gr.Button("Tahmin Et", elem_id="go", scale=2)

        # SAĞ PANEL
        with gr.Column(scale=1, min_width=400):
            result_html = gr.HTML(value=placeholder_html(), elem_id="res")

    predict_btn.click(fn=predict, inputs=[image_input, model_radio],
                      outputs=[result_html])
    image_input.change(fn=predict, inputs=[image_input, model_radio],
                       outputs=[result_html])

    gr.HTML('<div id="ft">Akademik amaçlıdır &middot; Tıbbi teşhis yerine geçmez</div>')

print("\nArayüz başlatılıyor...")
demo.launch(share=True)