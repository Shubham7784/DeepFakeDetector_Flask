import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

MODEL_PATH = "models/deepfake_detector.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.efficientnet_b0(pretrained=False)
    in_feats = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(in_feats, 2))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def predict(image_path, model):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    label = 'real' if probs[1] > probs[0] else 'fake'
    confidence = float(max(probs))
    return {'label': label, 'confidence': confidence}
