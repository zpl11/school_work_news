"""
真正的模型训练 - 使用全部 LIAR 数据集
"""
import json
import pickle
import time
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATASET_DIR = Path("./datasets/real_data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("\n" + "="*70)
    print("🤖 真正的模型训练 - 使用 LIAR 数据集")
    print("="*70 + "\n")
    
    # 加载数据集
    dataset_file = DATASET_DIR / "liar_dataset.json"
    
    print(f"📂 加载数据集...")
    start_time = time.time()
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    
    load_time = time.time() - start_time
    print(f"✅ 加载完成: {len(dataset)} 条 ({load_time:.2f}s)")
    print(f"   真实: {sum(1 for l in labels if l == 1)} 条")
    print(f"   虚假: {sum(1 for l in labels if l == 0)} 条")
    
    # 分割数据集
    print(f"\n📊 分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"   训练集: {len(X_train)} 条")
    print(f"   测试集: {len(X_test)} 条")
    
    # 创建管道
    print(f"\n🔧 创建模型管道...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])
    
    # 训练
    print(f"\n🤖 训练模型...")
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"✅ 训练完成 ({train_time:.2f}s)")
    
    # 评估
    print(f"\n📊 评估模型...")
    eval_start = time.time()
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    eval_time = time.time() - eval_start
    
    print(f"   准确率: {accuracy*100:.2f}%")
    print(f"   精确率: {precision*100:.2f}%")
    print(f"   召回率: {recall*100:.2f}%")
    print(f"   F1分数: {f1*100:.2f}%")
    print(f"   评估时间: {eval_time:.2f}s")
    
    # 保存模型
    print(f"\n💾 保存模型...")
    model_file = MODELS_DIR / "liar_trained_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ 模型已保存: {model_file}")
    print(f"   文件大小: {model_file.stat().st_size / (1024*1024):.2f} MB")
    
    # 测试
    print(f"\n🧪 测试模型:")
    test_cases = [
        ("The economic turnaround started at the end of my term.", 1),
        ("Says the Annies List political group supports third-trimester abortions on demand.", 0),
    ]
    
    for text, expected in test_cases:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        result = "✓" if pred == expected else "✗"
        print(f"   {result} {text[:50]}...")
        print(f"      预测: {'真实' if pred == 1 else '虚假'} (置信度: {max(proba)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

