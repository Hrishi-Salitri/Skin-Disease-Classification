import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from transformers import DeiTForImageClassification

# Set the page configuration
st.set_page_config(page_title="Skin Disease Classification", layout="wide")

# Define the list of classes
selected_classes = [
    "Psoriasis",
    "Fungal Infections",
    "Melanoma",
    "Nail Fungus",
    "Acne",
    "Warts",
    "Benign Tumors"
]

# Define severity categories
disease_classes = {
    "non_severe": ["Fungal Infections", "Nail Fungus", "Acne", "Warts"],
    "severe": ["Psoriasis", "Melanoma", "Benign Tumors"]
}

# Load models
@st.cache_resource
def load_resnet():
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(selected_classes))
    model.load_state_dict(torch.load("resnet34_skin_disease_balanced.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_inception():
    model = models.inception_v3(weights=None, aux_logits=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(selected_classes))
    model.load_state_dict(torch.load("inceptionv3_balanced_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(selected_classes))
    model.load_state_dict(torch.load("efficientnet_b1_balanced_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_deit():
    model = DeiTForImageClassification.from_pretrained(
        "facebook/deit-base-distilled-patch16-224", num_labels=len(selected_classes)
    )
    model.load_state_dict(torch.load("deit_finetuned_skin_disease_final_BalancedClasses.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

resnet_model = load_resnet()
inception_model = load_inception()
efficientnet_model = load_efficientnet()
deit_model = load_deit()

# Define transformations
transform_resnet_efficientnet_deit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict the class
def predict(image, model, transform):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        if hasattr(outputs, "logits"):  # For DeiT model
            outputs = outputs.logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
        _, predicted = torch.max(probabilities, 0)
        predicted_class = selected_classes[predicted.item()]
    return predicted_class, probabilities

# Function to determine severity
def determine_severity(predicted_class):
    if predicted_class in disease_classes["severe"]:
        return "Severe"
    return "Non-Severe"

# Sidebar
st.sidebar.title("Skin Disease Classifier")
st.sidebar.markdown("Upload an image to classify its severity and type using three advanced models.")
st.sidebar.info("Ensure the image is clear and focuses on the affected skin area for accurate predictions.")

st.sidebar.markdown(
    """
    ---
    **Disclaimer**

    This web application for classifying skin diseases was developed as part of a university project. 
    It is intended solely for educational and research purposes. The application uses a neural network model, 
    which may produce inaccurate or incomplete results. It is not a substitute for professional medical advice, 
    diagnosis, or treatment. Users should consult a qualified healthcare professional for any concerns about their 
    skin health. The creators of this application disclaim any liability for decisions made based on its output. 
    Use at your own risk.
    """
)

# Main layout
st.title("🌟 Skin Disease Classification")
st.markdown(
    """
    This application classifies skin diseases into **Severe** or **Non-Severe** categories.
    Additionally, it predicts the specific disease class using the following models:
    - ResNet-34
    - InceptionV3
    - EfficientNet-B1
    - DeiT (Vision Transformer)
    """
)

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.markdown("### Classification Results")
    st.info("The image is being processed. Below are the predictions from the models:")

    # DataFrame for results
    results = pd.DataFrame(columns=["Model", "Predicted Class", "Severity"] + selected_classes)

    # ResNet-34
    predicted_class_resnet, probabilities_resnet = predict(image, resnet_model, transform_resnet_efficientnet_deit)
    severity_resnet = determine_severity(predicted_class_resnet)
    resnet_row = {"Model": "ResNet-34", "Predicted Class": predicted_class_resnet, "Severity": severity_resnet}
    resnet_row.update({cls: f"{probabilities_resnet[i].item():.2%}" for i, cls in enumerate(selected_classes)})
    results = pd.concat([results, pd.DataFrame([resnet_row])], ignore_index=True)

    # InceptionV3
    predicted_class_inception, probabilities_inception = predict(image, inception_model, transform_inception)
    severity_inception = determine_severity(predicted_class_inception)
    inception_row = {"Model": "InceptionV3", "Predicted Class": predicted_class_inception, "Severity": severity_inception}
    inception_row.update({cls: f"{probabilities_inception[i].item():.2%}" for i, cls in enumerate(selected_classes)})
    results = pd.concat([results, pd.DataFrame([inception_row])], ignore_index=True)

    # EfficientNet-B0
    predicted_class_efficientnet, probabilities_efficientnet = predict(image, efficientnet_model, transform_resnet_efficientnet_deit)
    severity_efficientnet = determine_severity(predicted_class_efficientnet)
    efficientnet_row = {"Model": "EfficientNet-B1", "Predicted Class": predicted_class_efficientnet, "Severity": severity_efficientnet}
    efficientnet_row.update({cls: f"{probabilities_efficientnet[i].item():.2%}" for i, cls in enumerate(selected_classes)})
    results = pd.concat([results, pd.DataFrame([efficientnet_row])], ignore_index=True)

    # DeiT
    predicted_class_deit, probabilities_deit = predict(image, deit_model, transform_resnet_efficientnet_deit)
    severity_deit = determine_severity(predicted_class_deit)
    deit_row = {"Model": "DeiT", "Predicted Class": predicted_class_deit, "Severity": severity_deit}
    deit_row.update({cls: f"{probabilities_deit[i].item():.2%}" for i, cls in enumerate(selected_classes)})
    results = pd.concat([results, pd.DataFrame([deit_row])], ignore_index=True)

    # Display the results
    st.markdown("### Results Table")
    st.table(results.style.set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'tr:hover', 'props': [('background-color', '#f2f2f2')]}
    ]))

    st.success("Processing complete! Check the predictions in the table above.")
