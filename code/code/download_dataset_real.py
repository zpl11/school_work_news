"""
真正下载真实数据集 - 不生成任何数据
"""
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
import sys

DATASET_DIR = Path("./datasets/real_data")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_liar():
    """下载 LIAR 数据集 (12,800 条真实标注数据)"""
    print("\n📥 正在下载 LIAR 数据集...")
    print("   URL: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
    
    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    output = DATASET_DIR / "liar_dataset.zip"
    
    try:
        print("   连接中...")
        urllib.request.urlretrieve(url, output)
        print(f"✅ 下载完成: {output}")
        
        # 检查文件大小
        size = output.stat().st_size / (1024*1024)
        print(f"   文件大小: {size:.1f} MB")
        
        if size > 1:  # 真实文件应该 > 1MB
            print("✅ 数据集有效")
            return True
        else:
            print("❌ 文件太小，下载可能失败")
            return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_fakenewsnet():
    """下载 FakeNewsNet 数据集 (40,000+ 条)"""
    print("\n📥 正在下载 FakeNewsNet 数据集...")
    print("   URL: https://github.com/KaiDMML/FakeNewsNet/archive/refs/heads/master.zip")
    
    url = "https://github.com/KaiDMML/FakeNewsNet/archive/refs/heads/master.zip"
    output = DATASET_DIR / "fakenewsnet.zip"
    
    try:
        print("   连接中...")
        urllib.request.urlretrieve(url, output)
        print(f"✅ 下载完成: {output}")
        
        size = output.stat().st_size / (1024*1024)
        print(f"   文件大小: {size:.1f} MB")
        
        if size > 1:
            print("✅ 数据集有效")
            return True
        else:
            print("❌ 文件太小，下载可能失败")
            return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("📊 下载真实数据集 (不生成任何数据)")
    print("="*70)
    
    success = False
    
    # 尝试下载 LIAR
    if download_liar():
        success = True
    
    # 尝试下载 FakeNewsNet
    if download_fakenewsnet():
        success = True
    
    if success:
        print("\n" + "="*70)
        print("✅ 数据集下载完成！")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("❌ 无法下载数据集")
        print("   可能原因:")
        print("   1. 网络连接问题")
        print("   2. 服务器无法访问")
        print("   3. 防火墙限制")
        print("="*70 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()

