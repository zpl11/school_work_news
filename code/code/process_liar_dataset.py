"""
处理真实的 LIAR 数据集
"""
import json
from pathlib import Path

LIAR_DIR = Path("./liar")
DATASET_DIR = Path("./datasets/real_data")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def process_liar_tsv(tsv_file):
    """处理 LIAR TSV 文件"""
    dataset = []
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # 第2列是标签，第3列是声明文本
                label_str = parts[1].lower()
                text = parts[2]
                
                # 转换标签为二分类
                label_map = {
                    'true': 1,
                    'mostly-true': 1,
                    'half-true': 0,
                    'barely-true': 0,
                    'false': 0,
                    'pants-fire': 0
                }
                
                label = label_map.get(label_str, 0)
                
                if text.strip():
                    dataset.append({
                        "text": text,
                        "label": label,
                        "original_label": label_str
                    })
    
    return dataset

def main():
    print("\n" + "="*70)
    print("📊 处理真实 LIAR 数据集")
    print("="*70 + "\n")
    
    all_data = []
    
    # 处理训练集
    train_file = LIAR_DIR / "train.tsv"
    if train_file.exists():
        print(f"📂 处理 {train_file.name}...")
        train_data = process_liar_tsv(train_file)
        all_data.extend(train_data)
        print(f"   ✅ {len(train_data)} 条")
    
    # 处理验证集
    valid_file = LIAR_DIR / "valid.tsv"
    if valid_file.exists():
        print(f"📂 处理 {valid_file.name}...")
        valid_data = process_liar_tsv(valid_file)
        all_data.extend(valid_data)
        print(f"   ✅ {len(valid_data)} 条")
    
    # 处理测试集
    test_file = LIAR_DIR / "test.tsv"
    if test_file.exists():
        print(f"📂 处理 {test_file.name}...")
        test_data = process_liar_tsv(test_file)
        all_data.extend(test_data)
        print(f"   ✅ {len(test_data)} 条")
    
    # 保存
    output_file = DATASET_DIR / "liar_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    # 统计
    real_count = sum(1 for x in all_data if x['label'] == 1)
    fake_count = sum(1 for x in all_data if x['label'] == 0)
    
    print(f"\n✅ 数据集处理完成！")
    print(f"   总数: {len(all_data)} 条")
    print(f"   真实: {real_count} 条 ({real_count/len(all_data)*100:.1f}%)")
    print(f"   虚假: {fake_count} 条 ({fake_count/len(all_data)*100:.1f}%)")
    print(f"   保存: {output_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

