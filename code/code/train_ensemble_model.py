"""
使用集成学习模型训练 (RandomForest + SVM)
"""
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

DATASET_DIR = Path("./datasets/real_data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("\n" + "="*70)
    print("🤖 集成学习模型训练 (RandomForest + GradientBoosting)")
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
    
    # 分割数据
    print("\n📊 分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"   训练集: {len(X_train)} 条")
    print(f"   测试集: {len(X_test)} 条")
    
    # 模型 1: GradientBoosting
    print("\n🤖 模型 1: GradientBoosting 训练...")
    start = time.time()
    gb_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42))
    ])
    gb_model.fit(X_train, y_train)
    gb_time = time.time() - start

    y_pred_gb = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, y_pred_gb)
    print(f"✅ 完成 ({gb_time:.2f}s)")
    print(f"   准确率: {gb_acc*100:.2f}%")

    # 模型 2: SVM
    print("\n🤖 模型 2: SVM 训练...")
    start = time.time()
    svm_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LinearSVC(max_iter=2000, random_state=42))
    ])
    svm_model.fit(X_train, y_train)
    svm_time = time.time() - start
    
    y_pred_svm = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    print(f"✅ 完成 ({svm_time:.2f}s)")
    print(f"   准确率: {svm_acc*100:.2f}%")
    
    # 选择最好的模型
    print("\n📊 模型对比:")
    models = [
        ("GradientBoosting", gb_model, gb_acc, gb_time),
        ("SVM", svm_model, svm_acc, svm_time)
    ]
    
    for name, model, acc, train_time in models:
        print(f"   {name}: {acc*100:.2f}% ({train_time:.2f}s)")
    
    # 选择最好的
    best_model, best_name, best_acc = max(models, key=lambda x: x[2])[:3]
    print(f"\n🏆 最佳模型: {best_name} ({best_acc*100:.2f}%)")
    
    # 详细评估
    print(f"\n📊 详细评估 ({best_name}):")
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   准确率: {accuracy*100:.2f}%")
    print(f"   精确率: {precision*100:.2f}%")
    print(f"   召回率: {recall*100:.2f}%")
    print(f"   F1分数: {f1*100:.2f}%")
    
    # 保存最好的模型
    print(f"\n💾 保存模型...")
    model_file = MODELS_DIR / f"best_model_{best_name.lower()}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"✅ 模型已保存: {model_file}")
    print(f"   文件大小: {model_file.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n" + "="*70)
    print("✅ 集成学习训练完成！")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

