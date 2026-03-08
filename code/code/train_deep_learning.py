"""
使用 PyTorch 深度学习模型训练
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

DATASET_DIR = Path("./datasets/real_data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DeepTextClassifier(nn.Module):
    def __init__(self, input_size):
        super(DeepTextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def main():
    print("\n" + "="*70)
    print("🤖 深度学习模型训练 (PyTorch)")
    print("="*70 + "\n")
    
    # 加载数据
    print("📂 加载数据集...")
    dataset_file = DATASET_DIR / "liar_dataset.json"
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    
    print(f"✅ 加载完成: {len(dataset)} 条")
    
    # 特征提取
    print("\n🔧 特征提取 (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts).toarray()
    
    print(f"✅ 特征维度: {features.shape}")
    
    # 分割数据
    print("\n📊 分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"   训练集: {len(X_train)} 条")
    print(f"   测试集: {len(X_test)} 条")
    
    # 创建数据加载器
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 创建模型
    print("\n🔧 创建深度学习模型...")
    model = DeepTextClassifier(features.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    print("\n🤖 开始训练...")
    epochs = 5
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练
        model.train()
        total_loss = 0
        
        for i, (features, labels_batch) in enumerate(train_loader):
            features = features.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"   批次 {i+1}/{len(train_loader)}, 损失: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"   平均损失: {avg_loss:.4f}")
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels_batch in test_loader:
                features = features.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        accuracy = correct / total
        print(f"   准确率: {accuracy*100:.2f}%")
    
    # 保存模型
    print("\n💾 保存模型...")
    model_file = MODELS_DIR / "deep_learning_model.pth"
    torch.save(model.state_dict(), model_file)
    
    print(f"✅ 模型已保存: {model_file}")
    print(f"   文件大小: {model_file.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n" + "="*70)
    print("✅ 深度学习训练完成！")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

