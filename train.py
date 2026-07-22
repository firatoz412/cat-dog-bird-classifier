import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


from models.model import AnimalClassifierNet


 
def train_model():
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 15
    IMAGE_SIZE = (128,128)
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("eğitim cihazı:",device)
    
    
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3,contrast=0.2),
        transforms.ToTensor(),#piksel değerini 0-1 aralığındaki Tensor'e çevir..
        transforms.Normalize(mean=mean,std=std)
    ])
    
    
    
    DATA_DIR = "dataset"
    TRAIN_DIR =  os.path.join(DATA_DIR,"train")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"{TRAIN_DIR} klasörü bulunamadı.")
        return
    
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR,transform=train_transforms)
    
    #print(f"Sınıf İndeksleri: {train_dataset.class_to_idx}")
    #print(f"Toplam Train Resimleri: {len(train_dataset)}")
    
    
train_model()


