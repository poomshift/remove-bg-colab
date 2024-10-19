# Imports
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr

# Modify the import to avoid the registry issue
import sys
sys.path.append('.')
from models.birefnet import BiRefNet

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision(['high', 'highest'][0])
model = BiRefNet.from_pretrained('zhengpeng7/birefnet')
model.to(device)
model.eval()
print('BiRefNet is ready to use.')

# Input Data
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_background(image):
    w, h = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Resize the prediction to match the original image size
    pred_resized = transforms.Resize((h, w))(pred.unsqueeze(0)).squeeze()

    # Create masked image
    pred_pil = transforms.ToPILImage()(pred_resized)
    image_masked = image.copy()
    image_masked.putalpha(pred_pil)

    return image_masked

# Gradio Interface
iface = gr.Interface(
    fn=remove_background,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
    ],
    outputs=gr.Image(type="pil", label="Image with Background Removed"),
    title="Background Removal App",
    description="Upload an image to remove its background.",
)

# Launch the app and keep the cell running
iface.launch(share=True)
