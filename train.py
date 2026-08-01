import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights



#from models.model import AnimalClassifierNet

 
def train_model():
    BATCH_SIZE = 32
    LEARNING_RATE = 0.003
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
    
    
    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    
    
    DATA_DIR = "dataset"
    TRAIN_DIR =  os.path.join(DATA_DIR,"train")
    VAL_DIR = os.path.join(DATA_DIR,"val")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"{TRAIN_DIR} klasörü bulunamadı.")
        return
    
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR,transform=train_transforms)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    #print(f"Sınıf İndeksleri: {train_dataset.class_to_idx}")
    #print(f"Toplam Train Resimleri: {len(train_dataset)}")
    
    if not os.path.exists(VAL_DIR):
        print(f"{VAL_DIR} klasörü bulunamadı.")
        return

    val_dataset = datasets.ImageFolder(root=VAL_DIR,transform=val_transforms)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
        
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---", flush=True)
        model.train()#eğitim modu
        running_loss = 0.0
        correct = 0
        total = 0
        
        #train döngüsü
        for images, labels in train_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            #train hesaplamaları
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        train_loss = running_loss/total
        train_acc = (correct/total)*100  
    
        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        val_total = 0.0
    
        #validation döngüsü
        with torch.no_grad():
            for images,labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs,labels)
                
                #validation hesaplamaları
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs,1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss/val_total
        val_acc = (val_correct / val_total)*100
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.2f}")
        print(f"Val acc: {val_acc:.2f}")
        
    
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),"best_animal_model.pth")
            print(f"model kaydedildi: {best_acc}")
    
   
    torch.save(model.state_dict(),"animal_model_final.pth")
    

if __name__ == "__main__":
    train_model()


