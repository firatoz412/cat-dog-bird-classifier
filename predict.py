from scipy import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image



CLASS_NAMES = ['bird','cat','dog']

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class AnimalPredictor:
    def __init__(self,model_path="best_animal_model.pth"):
        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        self.model_path = model_path

        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)#0-1 oalasılık değerlerine dönüştür.
        top_prob, top_catid = torch.topk(probabilities, 1)
        predicted_class = CLASS_NAMES[top_catid[0].item()]
        confidence = top_prob[0].item()#float(...),en yüksek olasılık değeri
        #örnek:probabilities = tensor([0.96, 0.03, 0.01]) [cat,dog,bird]
        #top_prob:olasılığı en yüksek olan sınıfın olasılığı. top_prob = tensor([0.96])
        #top_catid: olasılığı en yüksek olan sınıfın indeksi. top_catid = tensor([0])
        
        return predicted_class, confidence,image
    

predictor = AnimalPredictor()
        
        