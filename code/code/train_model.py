"""
使用假数据集训练模型
"""
import json
import pickle
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("./datasets/real_data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class SimpleClassifier:
    """简单的文本分类器"""
    
    def __init__(self):
        self.real_keywords = set()
        self.fake_keywords = set()
    
    def fit(self, texts, labels):
        """训练"""
        print("🤖 训练模型...")
        
        real_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
        fake_texts = [texts[i] for i in range(len(texts)) if labels[i] == 0]
        
        # 提取真实新闻关键词
        real_words = []
        for text in real_texts:
            real_words.extend(text.split())
        
        real_counter = Counter(real_words)
        self.real_keywords = set([w for w, c in real_counter.most_common(50)])
        
        # 提取虚假新闻关键词
        fake_words = []
        for text in fake_texts:
            fake_words.extend(text.split())
        
        fake_counter = Counter(fake_words)
        self.fake_keywords = set([w for w, c in fake_counter.most_common(50)])
        
        print("✅ 训练完成")
    
    def predict(self, text):
        """预测"""
        real_count = sum(1 for w in self.real_keywords if w in text)
        fake_count = sum(1 for w in self.fake_keywords if w in text)
        
        if real_count > fake_count:
            return 1, 0.7
        else:
            return 0, 0.7
    
    def predict_proba(self, text):
        """预测概率"""
        label, score = self.predict(text)
        if label == 1:
            return [1 - score, score]
        else:
            return [score, 1 - score]

def main():
    print("\n" + "="*60)
    print("🤖 训练模型")
    print("="*60 + "\n")
    
    # 加载数据集
    dataset_file = DATASET_DIR / "fake_dataset.json"
    
    if not dataset_file.exists():
        print("❌ 数据集不存在，先创建...")
        import subprocess
        subprocess.run(["python", "create_fake_dataset.py"])
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"📊 加载数据集: {len(dataset)} 条")
    
    # 分离文本和标签
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    
    # 训练
    model = SimpleClassifier()
    model.fit(texts, labels)
    
    # 评估
    print("\n📊 评估模型...")
    correct = 0
    for text, label in zip(texts, labels):
        pred, _ = model.predict(text)
        if pred == label:
            correct += 1
    
    accuracy = correct / len(texts) * 100
    print(f"   准确率: {accuracy:.1f}%")
    
    # 保存
    model_file = MODELS_DIR / "model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ 模型已保存: {model_file}")
    
    # 测试
    print("\n🧪 测试:")
    test_cases = [
        ("新华社报道：国务院召开常务会议。", 1),
        ("惊爆！某明星秘密结婚生子！", 0),
    ]
    
    for text, expected in test_cases:
        pred, score = model.predict(text)
        result = "✓" if pred == expected else "✗"
        print(f"   {result} {text[:30]}... -> {'真实' if pred == 1 else '虚假'}")
    
    print("\n" + "="*60)
    print("✅ 完成！")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

