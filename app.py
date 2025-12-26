import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from model import DenseUNet

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 MRI Brain Tumor Segmentation")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DenseUNet().to(device)
model.load_state_dict(torch.load("tumor_model.pth", map_location=device))
model.eval()

uploaded = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (256, 256)) / 255.0

    tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        mask = model(tensor)[0][0].cpu().numpy()

    st.image(image, caption="Original MRI", use_container_width=True)
    st.image(mask, caption="Predicted Tumor Mask", use_container_width=True)
