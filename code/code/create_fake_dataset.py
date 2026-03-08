"""
创建假数据集用于训练
"""
import json
from pathlib import Path

DATASET_DIR = Path("./datasets/real_data")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 真实新闻样本
real_news = [
    "新华社报道：国务院召开常务会议，部署推进重点工作。",
    "央视新闻：今年上半年GDP增长5.3%，经济运行总体平稳。",
    "人民日报：我国科学家在量子计算领域取得重大突破。",
    "新华网：全国两会召开，代表委员围绕经济发展建言献策。",
    "中国日报：我国与多个国家签署合作协议，推进一带一路建设。",
    "光明日报：教育改革深入推进，学生综合素质不断提升。",
    "经济日报：工业生产稳步增长，制造业投资持续增加。",
    "科技日报：5G网络建设加快，覆盖范围不断扩大。",
    "中国新闻网：医疗卫生改革取得显著成效，人民健康水平提高。",
    "新华社：环保工作成效显著，空气质量明显改善。",
]

# 虚假新闻样本
fake_news = [
    "惊爆！某明星秘密结婚生子，隐瞒多年真相大白！",
    "震惊！某地发现外星人遗迹，政府隐瞒真相！",
    "独家爆料：某高官贪污数十亿，证据确凿！",
    "紧急通知：某疾病已在全球蔓延，政府隐瞒疫情！",
    "重磅消息：某公司破产，员工血本无归！",
    "曝光！某明星与富商秘密交易，内幕惊人！",
    "绝密：某国计划入侵我国，军方已做好准备！",
    "爆料：某官员与黑帮勾结，黑幕重重！",
    "突发：某地发生大规模骚乱，真相被隐瞒！",
    "独家：某企业偷税漏税数百亿，证据已掌握！",
]

# 创建数据集
dataset = []

# 添加真实新闻
for text in real_news:
    dataset.append({"text": text, "label": 1})

# 添加虚假新闻
for text in fake_news:
    dataset.append({"text": text, "label": 0})

# 保存
output_file = DATASET_DIR / "fake_dataset.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"✅ 创建假数据集: {output_file}")
print(f"   总数: {len(dataset)} 条")
print(f"   真实: {sum(1 for x in dataset if x['label'] == 1)} 条")
print(f"   虚假: {sum(1 for x in dataset if x['label'] == 0)} 条")

