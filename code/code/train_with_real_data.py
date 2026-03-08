"""
使用真实 LIAR 数据集训练模型
"""
import json
import pickle
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("./datasets/real_data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class TextClassifier:
    """文本分类器"""
    
    def __init__(self):
        self.real_keywords = set()
        self.fake_keywords = set()
    
    def fit(self, texts, labels):
        """训练"""
        print("🤖 训练模型...")
        
        real_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
        fake_texts = [texts[i] for i in range(len(texts)) if labels[i] == 0]
        
        print(f"   真实: {len(real_texts)} 条")
        print(f"   虚假: {len(fake_texts)} 条")
        
        # 提取真实新闻关键词
        real_words = []
        for text in real_texts[:1000]:  # 只用前1000条加速
            real_words.extend(text.lower().split())
        
        real_counter = Counter(real_words)
        self.real_keywords = set([w for w, c in real_counter.most_common(200) if len(w) > 2])
        
        # 提取虚假新闻关键词
        fake_words = []
        for text in fake_texts[:1000]:
            fake_words.extend(text.lower().split())
        
        fake_counter = Counter(fake_words)
        self.fake_keywords = set([w for w, c in fake_counter.most_common(200) if len(w) > 2])
        
        print("✅ 训练完成")
    
    def predict(self, text):
        """预测"""
        text_lower = text.lower()
        real_count = sum(1 for w in self.real_keywords if w in text_lower)
        fake_count = sum(1 for w in self.fake_keywords if w in text_lower)
        
        if real_count > fake_count:
            return 1, 0.6
        else:
            return 0, 0.6
    
    def predict_proba(self, text):
        """预测概率"""
        label, score = self.predict(text)
        if label == 1:
            return [1 - score, score]
        else:
            return [score, 1 - score]

def main():
    print("\n" + "="*70)
    print("🤖 使用真实 LIAR 数据集训练模型")
    print("="*70 + "\n")
    
    # 加载数据集
    dataset_file = DATASET_DIR / "liar_dataset.json"
    
    if not dataset_file.exists():
        print("❌ 数据集不存在")
        return
    
    print(f"📂 加载数据集...")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ 加载完成: {len(dataset)} 条")
    
    # 分离文本和标签
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    
    # 统计
    real_count = sum(1 for l in labels if l == 1)
    fake_count = sum(1 for l in labels if l == 0)
    print(f"   真实: {real_count} 条")
    print(f"   虚假: {fake_count} 条")
    
    # 训练
    model = TextClassifier()
    model.fit(texts, labels)
    
    # 评估（只用前1000条加速）
    print("\n📊 评估模型...")
    correct = 0
    for text, label in zip(texts[:1000], labels[:1000]):
        pred, _ = model.predict(text)
        if pred == label:
            correct += 1
    
    accuracy = correct / 1000 * 100
    print(f"   准确率: {accuracy:.1f}%")
    
    # 保存
    model_file = MODELS_DIR / "liar_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ 模型已保存: {model_file}")
    
    # 测试
    print("\n🧪 测试:")
    test_cases = [
        "The economic turnaround started at the end of my term.",
        "Says the Annies List political group supports third-trimester abortions on demand.",
    ]
    
    for text in test_cases:
        pred, score = model.predict(text)
        print(f"   {text[:50]}...")
        print(f"   -> {'真实' if pred == 1 else '虚假'}\n")
    
    print("="*70)
    print("✅ 完成！")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

