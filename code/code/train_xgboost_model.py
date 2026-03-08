"""
使用 XGBoost 训练 (高性能梯度提升)
"""
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pickle

DATASET_DIR = Path("./datasets/real_data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("\n" + "="*70)
    print("🤖 XGBoost 梯度提升模型训练")
    print("="*70 + "\n")
    
    # 加载数据
    print("📂 加载数据集...")
    dataset_file = DATASET_DIR / "liar_dataset.json"
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    
    print(f"✅ 加载完成: {len(dataset)} 条")
    print(f"   真实: {sum(1 for l in labels if l == 1)} 条")
    print(f"   虚假: {sum(1 for l in labels if l == 0)} 条")
    
    # 特征提取
    print("\n🔧 特征提取 (TF-IDF)...")
    start = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts)
    print(f"✅ 特征维度: {features.shape} ({time.time()-start:.2f}s)")
    
    # 分割数据
    print("\n📊 分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"   训练集: {len(X_train)} 条")
    print(f"   测试集: {len(X_test)} 条")
    
    # 创建 XGBoost 模型
    print("\n🔧 创建 XGBoost 模型...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    # 训练
    print("\n🤖 开始训练...")
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"✅ 训练完成 ({train_time:.2f}s)")
    
    # 评估
    print("\n📊 评估模型...")
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
    model_file = MODELS_DIR / "xgboost_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ 模型已保存: {model_file}")
    print(f"   文件大小: {model_file.stat().st_size / (1024*1024):.2f} MB")
    
    # 特征重要性
    print(f"\n📊 特征重要性 (Top 10):")
    feature_importance = model.feature_importances_
    top_indices = sorted(range(len(feature_importance)), 
                        key=lambda i: feature_importance[i], 
                        reverse=True)[:10]
    
    feature_names = vectorizer.get_feature_names_out()
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # 测试
    print(f"\n🧪 测试模型:")
    test_cases = [
        ("The economic turnaround started at the end of my term.", 1),
        ("Says the Annies List political group supports third-trimester abortions on demand.", 0),
    ]
    
    for text, expected in test_cases:
        text_features = vectorizer.transform([text])
        pred = model.predict(text_features)[0]
        proba = model.predict_proba(text_features)[0]
        result = "✓" if pred == expected else "✗"
        print(f"   {result} {text[:50]}...")
        print(f"      预测: {'真实' if pred == 1 else '虚假'} (置信度: {max(proba)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ XGBoost 训练完成！")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

