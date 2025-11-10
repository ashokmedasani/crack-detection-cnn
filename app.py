import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["Negative (No Crack)", "Positive (Crack Detected)"]

# ============================================================
# MODEL DEFINITIONS (MATCHING YOUR TRAINING CODE)
# ============================================================

class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        flatten_size = 64 * (IMG_SIZE // 4) * (IMG_SIZE // 4)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class CNN_BN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        flatten_size = 64 * (IMG_SIZE // 4) * (IMG_SIZE // 4)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ============================================================
# LOAD MODELS
# ============================================================

def load_model(path, model_type):
    if model_type == "baseline":
        model = CNN_Base()
    elif model_type == "bn":
        model = CNN_BN(dropout=0.0)
    else:
        model = CNN_BN(dropout=0.3)

    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


models = {
    "Baseline CNN": load_model("models/model_baseline.pt", "baseline"),
    "BatchNorm CNN": load_model("models/model_bn.pt", "bn"),
    "BN + Augmentation CNN": load_model("models/model_aug.pt", "aug")
}

# ============================================================
# TRANSFORM FOR INFERENCE (MATCHING TRAINING)
# ============================================================

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict(image, model_name):
    image = Image.fromarray(image).convert("RGB")
    img = infer_transform(image).unsqueeze(0).to(DEVICE)

    model = models[model_name]

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    return {
        LABELS[0]: float(probs[0]),
        LABELS[1]: float(probs[1])
    }


# ============================================================
# TOOLTIP TEXT
# ============================================================

info_baseline = (
    "Baseline CNN Architecture:\n"
    "- Conv2D(3→32), ReLU, MaxPool\n"
    "- Conv2D(32→64), ReLU, MaxPool\n"
    "- FC: 128 units → 2 classes\n"
    "- No BatchNorm, No Dropout\n"
)

info_bn = (
    "BatchNorm CNN Architecture:\n"
    "- Conv2D + BatchNorm(32), MaxPool\n"
    "- Conv2D + BatchNorm(64), MaxPool\n"
    "- FC: 256 units → 2 classes\n"
    "- Better stability, no dropout\n"
)

info_aug = (
    "BN + Augmentation CNN:\n"
    "- Same as BN model\n"
    "- Dropout = 0.3\n"
    "- Augmentations used:\n"
    "  * Horizontal Flip\n"
    "  * Rotation (±20°)\n"
    "  * ColorJitter\n"
)


# ============================================================
# CUSTOM RADIO WITH TOOLTIP
# ============================================================

def model_selector():
    return gr.Radio(
        ["Baseline CNN", "BatchNorm CNN", "BN + Augmentation CNN"],
        value="BN + Augmentation CNN",
        label="Choose Model",
        info="Baseline = No BN, BN = Stable, Aug = Best accuracy"
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(title="Concrete Crack Detection") as ui:

    gr.Markdown("## 🧱 Concrete Crack Detection using CNNs")
    gr.Markdown("Upload an image and compare predictions from 3 CNN models.")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Concrete Image")
        model_choice = model_selector()

    output = gr.Label(num_top_classes=2, label="Prediction")

    info_box = gr.Markdown("### ℹ️ Model Info appears here")

    def show_info(model_name):
        if model_name == "Baseline CNN":
            return info_baseline
        elif model_name == "BatchNorm CNN":
            return info_bn
        else:
            return info_aug

    model_choice.change(show_info, inputs=model_choice, outputs=info_box)

    gr.Button("Predict").click(predict, inputs=[image_input, model_choice], outputs=output)

ui.launch()
