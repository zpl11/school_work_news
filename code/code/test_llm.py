"""测试通义千问大模型API集成"""
import requests
import json

def test_api():
    """测试API连接"""
    api_key = "sk-28ff19e409b642ddbf1ed8afbed9c5ff"
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen-turbo",
        "messages": [
            {"role": "user", "content": "你好，请回复OK"}
        ],
        "max_tokens": 50
    }
    
    print("=" * 50)
    print("测试通义千问API连接...")
    print("=" * 50)
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=15)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
                print(f"API响应: {content}")
                print("✅ API连接测试成功!")
                return True
            else:
                print(f"API返回异常: {result}")
                return False
        else:
            print(f"API错误: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

def test_llm_analyzer():
    """测试LLM分析器"""
    print("\n" + "=" * 50)
    print("测试LLM分析器...")
    print("=" * 50)
    
    try:
        from analysis_engine import llm_analyzer
        
        # 测试一个标题党新闻
        title = "震惊！99%的人不知道的秘密"
        content = "今天天气很好，适合出门散步。公园里的花开了，很漂亮。"
        
        print(f"测试标题: {title}")
        print(f"测试内容: {content}")
        print("-" * 30)
        
        result = llm_analyzer.analyze(title, content)
        
        print(f"分析结果:")
        print(f"  - 可信度评分: {result.get('credibility_score', 'N/A')}")
        print(f"  - 可信度标签: {result.get('credibility_label', 'N/A')}")
        print(f"  - 标题党检测: {result.get('clickbait_detected', 'N/A')}")
        print(f"  - 分析理由: {result.get('analysis_reason', 'N/A')[:100]}")
        print(f"  - LLM可用: {result.get('llm_available', False)}")
        
        if result.get("llm_available"):
            print("✅ LLM分析器测试成功!")
            return True
        else:
            print(f"⚠️ LLM不可用: {result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ LLM分析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_analyzer():
    """测试完整的文本分析流程"""
    print("\n" + "=" * 50)
    print("测试完整文本分析流程（含LLM）...")
    print("=" * 50)
    
    try:
        from analysis_engine import TextAnalyzer
        
        analyzer = TextAnalyzer()
        
        title = "科学家发现新型清洁能源技术"
        content = """
        近日，中国科学院的研究团队在清洁能源领域取得重大突破。
        该团队成功研发出一种新型太阳能电池，转换效率达到30%以上。
        这项技术有望在未来5年内实现商业化应用，届时将大幅降低清洁能源成本。
        研究成果已发表在《自然》杂志上。
        """
        
        print(f"测试标题: {title}")
        print(f"测试内容: {content[:50]}...")
        print("-" * 30)
        
        result = analyzer.analyze(content, title, use_llm=True)
        
        print(f"分析结果:")
        print(f"  - 综合评分: {result.get('score', 'N/A')}")
        print(f"  - 可信度: {result.get('credibility', 'N/A')}")
        
        if "llm_analysis" in result:
            llm = result["llm_analysis"]
            print(f"  - LLM评分: {llm.get('score', 'N/A')}")
            print(f"  - LLM标签: {llm.get('label', 'N/A')}")
            print(f"  - LLM分析: {llm.get('analysis_reason', 'N/A')[:80]}")
            print("✅ 完整分析测试成功!")
        else:
            print("⚠️ LLM分析结果未包含在返回中")
            
        return True
        
    except Exception as e:
        print(f"❌ 文本分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🚀 开始测试通义千问大模型集成\n")
    
    # 测试1: API连接
    api_ok = test_api()
    
    # 测试2: LLM分析器
    if api_ok:
        llm_ok = test_llm_analyzer()
    
    # 测试3: 完整流程
    if api_ok:
        test_text_analyzer()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)

