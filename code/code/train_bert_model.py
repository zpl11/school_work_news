"""
使用 BERT 深度学习模型训练
"""
import json
import torch
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = Path("./datasets/real_data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class NewsDataset(Dataset):
    """新闻数据集"""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"   批次 {i+1}/{len(dataloader)}")

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    print("\n" + "="*70)
    print("🤖 BERT 深度学习模型训练")
    print("="*70 + "\n")
    
    # 加载数据
    print("📂 加载数据集...")
    dataset_file = DATASET_DIR / "liar_dataset.json"
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    
    print(f"✅ 加载完成: {len(dataset)} 条")
    
    # 分割数据
    print("\n📊 分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"   训练集: {len(X_train)} 条")
    print(f"   测试集: {len(X_test)} 条")
    
    # 加载 BERT
    print("\n🔧 加载 BERT 模型...")
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # 创建数据加载器
    print("📦 创建数据加载器...")
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 训练
    print("\n🤖 开始训练...")
    epochs = 3
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_time = time.time() - train_start
        
        eval_start = time.time()
        accuracy = evaluate(model, test_loader, device)
        eval_time = time.time() - eval_start
        
        print(f"   损失: {train_loss:.4f} ({train_time:.2f}s)")
        print(f"   准确率: {accuracy*100:.2f}% ({eval_time:.2f}s)")
    
    # 保存模型
    print("\n💾 保存模型...")
    model_path = MODELS_DIR / "bert_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"✅ 模型已保存: {model_path}")
    
    print("\n" + "="*70)
    print("✅ BERT 训练完成！")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

