"""
最终模型训练 - SVM + Naive Bayes (快速高效)
"""
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

DATASET_DIR = Path("./datasets/real_data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("\n" + "="*70)
    print("🤖 最终模型训练 (SVM + Naive Bayes)")
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
    
    # 模型 1: SVM
    print("\n🤖 模型 1: SVM 训练...")
    start = time.time()
    svm_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LinearSVC(max_iter=2000, random_state=42, dual=False))
    ])
    svm_model.fit(X_train, y_train)
    svm_time = time.time() - start
    
    y_pred_svm = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_prec = precision_score(y_test, y_pred_svm)
    svm_rec = recall_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)
    
    print(f"✅ 完成 ({svm_time:.2f}s)")
    print(f"   准确率: {svm_acc*100:.2f}%")
    print(f"   精确率: {svm_prec*100:.2f}%")
    print(f"   召回率: {svm_rec*100:.2f}%")
    print(f"   F1分数: {svm_f1*100:.2f}%")
    
    # 模型 2: Naive Bayes
    print("\n🤖 模型 2: Naive Bayes 训练...")
    start = time.time()
    nb_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])
    nb_model.fit(X_train, y_train)
    nb_time = time.time() - start
    
    y_pred_nb = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_prec = precision_score(y_test, y_pred_nb)
    nb_rec = recall_score(y_test, y_pred_nb)
    nb_f1 = f1_score(y_test, y_pred_nb)
    
    print(f"✅ 完成 ({nb_time:.2f}s)")
    print(f"   准确率: {nb_acc*100:.2f}%")
    print(f"   精确率: {nb_prec*100:.2f}%")
    print(f"   召回率: {nb_rec*100:.2f}%")
    print(f"   F1分数: {nb_f1*100:.2f}%")
    
    # 选择最好的模型
    print("\n📊 模型对比:")
    models = [
        ("SVM", svm_model, svm_acc, svm_time),
        ("Naive Bayes", nb_model, nb_acc, nb_time)
    ]
    
    for name, model, acc, train_time in models:
        print(f"   {name}: {acc*100:.2f}% ({train_time:.2f}s)")
    
    # 选择最好的
    best_name, best_model, best_acc, _ = max(models, key=lambda x: x[2])
    print(f"\n🏆 最佳模型: {best_name} ({best_acc*100:.2f}%)")
    
    # 保存最好的模型
    print(f"\n💾 保存模型...")
    model_file = MODELS_DIR / f"best_model_{best_name.lower().replace(' ', '_')}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"✅ 模型已保存: {model_file}")
    print(f"   文件大小: {model_file.stat().st_size / (1024*1024):.2f} MB")
    
    # 测试
    print(f"\n🧪 测试模型:")
    test_cases = [
        ("The economic turnaround started at the end of my term.", 1),
        ("Says the Annies List political group supports third-trimester abortions on demand.", 0),
    ]
    
    for text, expected in test_cases:
        pred = best_model.predict([text])[0]
        proba = best_model.predict_proba([text])[0]
        result = "✓" if pred == expected else "✗"
        print(f"   {result} {text[:50]}...")
        print(f"      预测: {'真实' if pred == 1 else '虚假'} (置信度: {max(proba)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ 模型训练完成！")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

