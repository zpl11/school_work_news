import json
import time
import requests
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import pickle
from pathlib import Path
from functools import lru_cache

# 加载训练好的模型
print("📥 加载训练好的新闻真实性分类模型...")
try:
    model_path = Path(__file__).parent / "models" / "best_model_naive_bayes.pkl"
    with open(model_path, 'rb') as f:
        trained_model = pickle.load(f)
    print("✅ 训练模型加载成功")
    TRAINED_MODEL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ 训练模型加载失败: {e}")
    TRAINED_MODEL_AVAILABLE = False
    trained_model = None

# BERT模型已禁用，使用训练好的模型进行分类
BERT_AVAILABLE = False
tokenizer = None
model = None

print("✅ 使用训练好的模型进行文本分类")

# ==================== 真实性判定标准 ====================
# 评分阈值定义（满分100分）
# 三分类标准：真实、可疑、虚假
CREDIBILITY_THRESHOLDS = {
    "real": 70,        # >= 70分：真实新闻
    "suspicious": 40,  # 40-69分：可疑新闻
    "fake": 0          # < 40分：虚假新闻
}

def get_credibility_label(score: float) -> str:
    """根据评分获取可信度标签（三分类）"""
    if score >= CREDIBILITY_THRESHOLDS["real"]:
        return "真实"
    elif score >= CREDIBILITY_THRESHOLDS["suspicious"]:
        return "可疑"
    else:
        return "虚假"

def get_credibility_description(score: float) -> str:
    """获取可信度详细描述"""
    if score >= 85:
        return "高度可信 - 内容真实可靠，各项指标均表现良好"
    elif score >= 70:
        return "基本可信 - 内容较为可靠，建议进一步核实"
    elif score >= 55:
        return "存疑 - 部分内容可能存在问题，需要谨慎对待"
    elif score >= 40:
        return "可疑 - 内容可信度较低，建议多方求证"
    elif score >= 25:
        return "高度可疑 - 内容很可能不实，请勿轻信"
    else:
        return "虚假 - 内容极不可信，存在明显虚假特征"

# ==================== 一致性检测类 ====================
class ConsistencyChecker:
    """一致性检测器 - 检测标题与正文、图片与文本的一致性"""

    def __init__(self):
        # 常见的标题党关键词
        self.clickbait_keywords = [
            '震惊', '惊呆', '竟然', '居然', '万万没想到', '太可怕了',
            '速看', '紧急', '刚刚', '突发', '重磅', '爆料',
            '99%的人不知道', '看完惊呆了', '不转不是中国人'
        ]

    def check_title_content_consistency(self, title: str, content: str) -> Dict[str, Any]:
        """检测标题和正文的一致性"""
        if not title or not content:
            return {
                "consistency_score": 0.5,
                "is_consistent": True,
                "reason": "标题或正文为空，无法检测"
            }

        title_lower = title.lower()
        content_lower = content.lower()

        # 1. 关键词重叠度检测
        title_words = set(self._tokenize(title))
        content_words = set(self._tokenize(content))

        if len(title_words) == 0:
            overlap_ratio = 0
        else:
            overlap = title_words & content_words
            overlap_ratio = len(overlap) / len(title_words)

        # 2. 标题党检测
        clickbait_score = self._detect_clickbait(title)

        # 3. 语义相关性（简化版：检查标题中的实体是否出现在正文中）
        semantic_score = self._check_semantic_relevance(title, content)

        # 综合一致性评分
        consistency_score = (overlap_ratio * 0.4 + (1 - clickbait_score) * 0.3 + semantic_score * 0.3)

        # 判定是否一致
        is_consistent = consistency_score >= 0.4

        # 生成原因说明
        reasons = []
        if overlap_ratio < 0.3:
            reasons.append("标题与正文关键词重叠度低")
        if clickbait_score > 0.5:
            reasons.append("标题存在标题党特征")
        if semantic_score < 0.3:
            reasons.append("标题与正文语义相关性低")

        return {
            "consistency_score": round(consistency_score, 3),
            "is_consistent": is_consistent,
            "overlap_ratio": round(overlap_ratio, 3),
            "clickbait_score": round(clickbait_score, 3),
            "semantic_score": round(semantic_score, 3),
            "reason": "；".join(reasons) if reasons else "标题与正文一致性良好"
        }

    def check_image_text_consistency(self, image_ocr_text: str, news_content: str) -> Dict[str, Any]:
        """检测图片内容（OCR文字）与新闻文本的一致性"""
        if not image_ocr_text or image_ocr_text in ["未检测到文字", "OCR不可用"]:
            return {
                "consistency_score": 0.7,  # 无法检测时给予中等分数
                "is_consistent": True,
                "reason": "图片中未检测到文字，无法进行一致性检测"
            }

        if not news_content:
            return {
                "consistency_score": 0.5,
                "is_consistent": True,
                "reason": "新闻正文为空，无法检测"
            }

        # 提取图片中的关键词
        image_words = set(self._tokenize(image_ocr_text))
        content_words = set(self._tokenize(news_content))

        if len(image_words) == 0:
            return {
                "consistency_score": 0.7,
                "is_consistent": True,
                "reason": "图片文字过少，无法有效检测"
            }

        # 计算重叠度
        overlap = image_words & content_words
        overlap_ratio = len(overlap) / len(image_words)

        # 一致性评分
        consistency_score = min(overlap_ratio * 1.5, 1.0)  # 放大重叠度的影响
        is_consistent = consistency_score >= 0.3

        reason = "图片内容与文本一致" if is_consistent else "图片内容与文本不一致，可能存在配图错误"

        return {
            "consistency_score": round(consistency_score, 3),
            "is_consistent": is_consistent,
            "overlap_ratio": round(overlap_ratio, 3),
            "reason": reason
        }

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（中英文混合）"""
        import re
        # 移除标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分词
        words = text.split()
        # 过滤停用词和短词
        stopwords = {'的', '了', '是', '在', '和', '与', '或', 'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        return [w for w in words if len(w) > 1 and w.lower() not in stopwords]

    def _detect_clickbait(self, title: str) -> float:
        """检测标题党"""
        score = 0
        for keyword in self.clickbait_keywords:
            if keyword in title:
                score += 0.2
        return min(score, 1.0)

    def _check_semantic_relevance(self, title: str, content: str) -> float:
        """检查语义相关性（简化版）"""
        # 提取标题中的数字、专有名词等
        import re

        # 检查数字一致性
        title_numbers = set(re.findall(r'\d+', title))
        content_numbers = set(re.findall(r'\d+', content))

        if title_numbers:
            number_match = len(title_numbers & content_numbers) / len(title_numbers)
        else:
            number_match = 1.0

        # 检查标题关键词在正文中的出现频率
        title_words = self._tokenize(title)
        if not title_words:
            return 0.5

        match_count = sum(1 for w in title_words if w in content)
        keyword_match = match_count / len(title_words)

        return (number_match * 0.4 + keyword_match * 0.6)


# 创建一致性检测器实例
consistency_checker = ConsistencyChecker()


# ==================== 大语言模型分析类 ====================
class LLMAnalyzer:
    """通义千问大模型辅助分析器"""

    def __init__(self):
        self.api_key = "sk-28ff19e409b642ddbf1ed8afbed9c5ff"
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = "qwen-turbo"  # 使用qwen-turbo，性价比高
        self.timeout = 15  # 超时时间（秒）
        self.enabled = True  # 是否启用LLM分析
        self.last_call_time = 0
        self.min_interval = 1  # 最小调用间隔（秒），防止频繁调用

        print("🤖 通义千问大模型分析器已初始化")

    def analyze(self, title: str, content: str) -> Dict[str, Any]:
        """
        使用通义千问分析新闻真实性
        返回：评分、分析理由、标题党检测、事实核查建议
        """
        if not self.enabled:
            return self._get_default_result("LLM分析已禁用")

        if not content or len(content) < 10:
            return self._get_default_result("内容过短，无法分析")

        # 频率限制
        current_time = time.time()
        if current_time - self.last_call_time < self.min_interval:
            time.sleep(self.min_interval - (current_time - self.last_call_time))

        try:
            # 构建分析prompt
            prompt = self._build_prompt(title, content)

            # 调用API
            response = self._call_api(prompt)

            # 解析响应
            result = self._parse_response(response)

            self.last_call_time = time.time()
            print(f"✅ LLM分析完成: 可信度={result['credibility_score']}, 标签={result['credibility_label']}")

            return result

        except requests.Timeout:
            print("⚠️ LLM API调用超时，使用降级方案")
            return self._get_default_result("API调用超时")
        except requests.RequestException as e:
            print(f"⚠️ LLM API调用失败: {e}")
            return self._get_default_result(f"API调用失败: {str(e)}")
        except Exception as e:
            print(f"⚠️ LLM分析异常: {e}")
            return self._get_default_result(f"分析异常: {str(e)}")

    def _build_prompt(self, title: str, content: str) -> str:
        """构建分析prompt"""
        # 截断过长内容
        max_content_length = 2000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "...(内容已截断)"

        prompt = f"""你是一个专业的新闻真实性核查专家。请分析以下新闻的真实性，并给出评估。

【新闻标题】
{title or "无标题"}

【新闻内容】
{content}

请从以下几个维度进行分析，并以JSON格式返回结果：

1. **可信度评分** (credibility_score): 0-100分，表示新闻的可信程度
   - 90-100: 高度可信，有明确来源和事实依据
   - 70-89: 基本可信，内容合理但需进一步核实
   - 40-69: 存疑，部分内容可能不实
   - 0-39: 高度可疑或虚假

2. **可信度标签** (credibility_label): "真实"/"可疑"/"虚假"

3. **标题党检测** (clickbait_detected): true/false
   - 检测标题是否夸大、耸人听闻、与内容不符

4. **逻辑一致性** (logic_consistency): 0-100分
   - 检测内容是否存在自相矛盾

5. **分析理由** (analysis_reason): 简要说明判断依据（100字以内）

6. **事实核查建议** (fact_check_suggestions): 列出需要核实的关键信息点（数组格式）

请严格按照以下JSON格式返回，不要包含其他内容：
```json
{{
    "credibility_score": 数字,
    "credibility_label": "标签",
    "clickbait_detected": 布尔值,
    "logic_consistency": 数字,
    "analysis_reason": "分析理由",
    "fact_check_suggestions": ["建议1", "建议2"]
}}
```"""
        return prompt

    def _call_api(self, prompt: str) -> str:
        """调用通义千问API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的新闻真实性核查专家，擅长分析新闻内容的可信度。请始终以JSON格式返回分析结果。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,  # 降低随机性，使结果更稳定
            "max_tokens": 800
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=self.timeout
        )

        response.raise_for_status()
        result = response.json()

        # 提取回复内容
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError("API返回格式异常")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试提取JSON部分
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                # 验证并规范化数据
                result = {
                    "credibility_score": self._normalize_score(data.get("credibility_score", 50)),
                    "credibility_label": data.get("credibility_label", "可疑"),
                    "clickbait_detected": bool(data.get("clickbait_detected", False)),
                    "logic_consistency": self._normalize_score(data.get("logic_consistency", 50)),
                    "analysis_reason": str(data.get("analysis_reason", "无法获取分析理由"))[:200],
                    "fact_check_suggestions": data.get("fact_check_suggestions", []),
                    "llm_available": True,
                    "raw_response": response[:500]  # 保留部分原始响应用于调试
                }

                return result
            else:
                raise ValueError("响应中未找到JSON")

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            # 尝试从文本中提取关键信息
            return self._extract_from_text(response)

    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """从非JSON文本中提取分析结果"""
        result = self._get_default_result("JSON解析失败，从文本提取")

        # 尝试提取评分
        import re
        score_match = re.search(r'(\d{1,3})\s*分', text)
        if score_match:
            score = int(score_match.group(1))
            result["credibility_score"] = min(100, max(0, score))

        # 检测关键词判断可信度
        if "虚假" in text or "不实" in text or "谣言" in text:
            result["credibility_label"] = "虚假"
            result["credibility_score"] = min(result["credibility_score"], 35)
        elif "可疑" in text or "存疑" in text:
            result["credibility_label"] = "可疑"
        elif "真实" in text or "可信" in text:
            result["credibility_label"] = "真实"

        # 检测标题党
        if "标题党" in text or "夸大" in text or "耸人听闻" in text:
            result["clickbait_detected"] = True

        result["analysis_reason"] = text[:200] if len(text) > 200 else text
        result["llm_available"] = True

        return result

    def _normalize_score(self, score) -> float:
        """规范化评分到0-100"""
        try:
            score = float(score)
            return min(100, max(0, score))
        except:
            return 50.0

    def _get_default_result(self, reason: str) -> Dict[str, Any]:
        """返回默认结果（降级方案）"""
        return {
            "credibility_score": 50,
            "credibility_label": "未知",
            "clickbait_detected": False,
            "logic_consistency": 50,
            "analysis_reason": reason,
            "fact_check_suggestions": [],
            "llm_available": False,
            "error": reason
        }

    def set_enabled(self, enabled: bool):
        """启用或禁用LLM分析"""
        self.enabled = enabled
        print(f"🤖 LLM分析已{'启用' if enabled else '禁用'}")


# 创建LLM分析器实例
llm_analyzer = LLMAnalyzer()


class TextAnalyzer:
    """文本分析 - 使用训练好的模型 + LLM辅助分析"""

    def __init__(self):
        self.use_llm = True  # 是否使用LLM辅助分析
        self.llm_weight = 0.25  # LLM分析在综合评分中的权重（25%）

    def analyze(self, text: str, title: str = None, use_llm: bool = True) -> Dict[str, Any]:
        length = len(text)
        words = len(text.split())

        # ==================== 1. 训练模型预测 ====================
        credibility_score = 0.5
        credibility_label = "未知"

        if TRAINED_MODEL_AVAILABLE and len(text) > 5:
            try:
                prediction = trained_model.predict([text])[0]
                probabilities = trained_model.predict_proba([text])[0]
                credibility_score = probabilities[1]
                THRESHOLD = 0.34  # 超过34%就判定为真实
                credibility_label = "真实" if credibility_score >= THRESHOLD else "虚假"
                print(f"🤖 模型预测: {credibility_label} (真实概率: {credibility_score*100:.1f}%, 阈值: {THRESHOLD*100:.0f}%)")
            except Exception as e:
                print(f"⚠️ 模型预测错误: {e}")
                credibility_score = 0.5
                credibility_label = "未知"

        # ==================== 2. LLM辅助分析 ====================
        llm_result = None
        llm_score = 0.5
        if use_llm and self.use_llm and len(text) > 20:
            try:
                llm_result = llm_analyzer.analyze(title or "", text)
                if llm_result.get("llm_available", False):
                    llm_score = llm_result["credibility_score"] / 100  # 转换为0-1
                    print(f"🧠 LLM分析: {llm_result['credibility_label']} ({llm_result['credibility_score']}分)")
            except Exception as e:
                print(f"⚠️ LLM分析失败: {e}")
                llm_result = None

        # ==================== 3. BERT情感分析 ====================
        if BERT_AVAILABLE and len(text) > 5:
            try:
                inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    sentiment_score = float(probabilities[0][1])
            except Exception as e:
                print(f"BERT分析错误: {e}")
                sentiment_score = 0.5
        else:
            sentiment_score = 0.5

        # ==================== 4. 其他分析 ====================
        contradiction_score = self._detect_contradiction(text)
        length_score = min(length / 1000, 1.0)
        keyword_score = self._extract_keywords(text)

        # 标题-正文一致性检测
        title_consistency = None
        title_consistency_score = 1.0
        if title:
            title_consistency = consistency_checker.check_title_content_consistency(title, text)
            title_consistency_score = title_consistency["consistency_score"]
            if not title_consistency["is_consistent"]:
                print(f"⚠️ 标题-正文不一致: {title_consistency['reason']}")

        # ==================== 5. 综合评分计算 ====================
        # 如果LLM可用，调整权重分配
        if llm_result and llm_result.get("llm_available", False):
            # 有LLM时: 模型30% + LLM25% + 标题一致性15% + 情感10% + 长度10% + 关键词5% + 矛盾5%
            overall = (
                credibility_score * 0.30 +
                llm_score * 0.25 +
                title_consistency_score * 0.15 +
                sentiment_score * 0.10 +
                length_score * 0.10 +
                keyword_score * 0.05 +
                (1 - contradiction_score) * 0.05
            ) * 100
        else:
            # 无LLM时: 模型40% + 标题一致性20% + 情感15% + 长度10% + 关键词10% + 矛盾5%
            overall = (
                credibility_score * 0.40 +
                title_consistency_score * 0.20 +
                sentiment_score * 0.15 +
                length_score * 0.10 +
                keyword_score * 0.10 +
                (1 - contradiction_score) * 0.05
            ) * 100

        # ==================== 6. 构建返回结果 ====================
        result = {
            "length": length,
            "words": words,
            "credibility": credibility_label,
            "credibility_score": round(credibility_score, 3),
            "sentiment": round(sentiment_score, 3),
            "contradiction": round(contradiction_score, 3),
            "length_score": round(length_score, 3),
            "keyword_score": round(keyword_score, 3),
            "title_consistency_score": round(title_consistency_score, 3),
            "score": round(overall, 2)
        }

        # 添加一致性详情
        if title_consistency:
            result["title_consistency"] = title_consistency

        # 添加LLM分析结果
        if llm_result:
            result["llm_analysis"] = {
                "available": llm_result.get("llm_available", False),
                "score": llm_result.get("credibility_score", 50),
                "label": llm_result.get("credibility_label", "未知"),
                "clickbait_detected": llm_result.get("clickbait_detected", False),
                "logic_consistency": llm_result.get("logic_consistency", 50),
                "analysis_reason": llm_result.get("analysis_reason", ""),
                "fact_check_suggestions": llm_result.get("fact_check_suggestions", [])
            }

        return result

    def _detect_contradiction(self, text: str) -> float:
        """检测文本中的矛盾 - BERT语义分析"""
        contradiction_pairs = [
            ('是', '不是'),
            ('yes', 'no'),
            ('true', 'false'),
            ('好', '坏'),
            ('增加', '减少'),
            ('支持', '反对'),
            ('同意', '不同意'),
        ]

        text_lower = text.lower()
        contradiction_count = 0

        for word1, word2 in contradiction_pairs:
            if word1 in text_lower and word2 in text_lower:
                contradiction_count += 1

        return min(contradiction_count * 0.15, 1.0)

    def _extract_keywords(self, text: str) -> float:
        """提取关键词评分"""
        # 检查是否包含新闻关键词
        news_keywords = ['报道', '新闻', '事件', '发生', '表示', '称', '指出', '认为', '表态', '声明']
        keyword_count = sum(1 for kw in news_keywords if kw in text)

        # 关键词越多，评分越高
        return min(keyword_count / 5, 1.0)

class ImageAnalyzer:
    """图像分析 - 使用OpenCV和深度学习"""
    def analyze(self, file_path: str, news_content: str = None) -> Dict[str, Any]:
        try:
            from PIL import Image
            import cv2

            img = Image.open(file_path)
            width, height = img.size

            # 使用OpenCV分析图像质量
            cv_img = cv2.imread(file_path)
            if cv_img is None:
                return {"error": "无法读取图像", "score": 0}

            # 计算图像清晰度（使用Laplacian方差）
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 清晰度评分（方差越大越清晰）
            quality_score = min(laplacian_var / 500, 1.0)

            # 分辨率评分
            resolution_score = min((width * height) / 1000000, 1.0)

            # 篡改检测 - 检查图像的一致性
            tampering_score = self._detect_tampering(cv_img)

            # OCR文字提取
            ocr_text = self._extract_text_ocr(file_path)

            # 图片-文本一致性检测
            image_text_consistency = None
            consistency_score = 1.0  # 默认一致
            if news_content and ocr_text and ocr_text not in ["未检测到文字", "OCR不可用"]:
                image_text_consistency = consistency_checker.check_image_text_consistency(ocr_text, news_content)
                consistency_score = image_text_consistency["consistency_score"]
                if not image_text_consistency["is_consistent"]:
                    print(f"⚠️ 图片-文本不一致: {image_text_consistency['reason']}")

            # 综合评分 = 质量30% + 分辨率20% + 篡改检测25% + 图文一致性25%
            overall = (
                quality_score * 0.30 +
                resolution_score * 0.20 +
                (1 - tampering_score) * 0.25 +
                consistency_score * 0.25
            ) * 100

            result = {
                "width": width,
                "height": height,
                "quality": round(quality_score, 3),
                "resolution": round(resolution_score, 3),
                "tampering": round(tampering_score, 3),
                "ocr_text": ocr_text,
                "image_text_consistency_score": round(consistency_score, 3),
                "score": round(overall, 2)
            }

            if image_text_consistency:
                result["image_text_consistency"] = image_text_consistency

            return result
        except Exception as e:
            return {"error": f"图像分析失败: {str(e)}", "score": 0}

    def _detect_tampering(self, img) -> float:
        """检测图像篡改迹象"""
        try:
            import cv2
            # 使用边缘检测
            edges = cv2.Canny(img, 100, 200)
            edge_ratio = np.sum(edges > 0) / edges.size

            # 边缘过多可能表示篡改
            tampering_score = min(edge_ratio * 2, 1.0)
            return tampering_score
        except:
            return 0.1

    def _extract_text_ocr(self, file_path: str) -> str:
        """使用OCR提取图像中的文字"""
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            return text[:200] if text else "未检测到文字"
        except:
            return "OCR不可用"

    def compare_images(self, file_path1: str, file_path2: str) -> float:
        """比较两张图像的相似度"""
        try:
            import cv2
            from skimage.metrics import structural_similarity as ssim

            img1 = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                return 0.0

            # 调整大小以便比较
            h, w = img1.shape
            img2 = cv2.resize(img2, (w, h))

            # 计算结构相似度
            similarity = ssim(img1, img2)
            return round(similarity, 3)
        except:
            return 0.0

class VideoAnalyzer:
    """视频分析 - 关键帧提取、音频分离、内容一致性分析"""
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            import cv2

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return {"error": "无法读取视频", "score": 0}

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 提取关键帧
            keyframes = self._extract_keyframes(cap, frame_count)

            # 内容一致性分析
            consistency_score = self._analyze_consistency(cap, frame_count)

            # 视频质量评分
            quality_score = min((width * height) / 1000000, 1.0)

            cap.release()

            overall = (quality_score * 0.4 + consistency_score * 0.6) * 100

            return {
                "fps": fps,
                "frame_count": frame_count,
                "resolution": f"{width}x{height}",
                "quality": round(quality_score, 3),
                "consistency": round(consistency_score, 3),
                "keyframes_count": len(keyframes),
                "score": round(overall, 2)
            }
        except Exception as e:
            return {"error": f"视频分析失败: {str(e)}", "score": 0}

    def _extract_keyframes(self, cap, frame_count: int, num_keyframes: int = 5) -> list:
        """提取关键帧"""
        import cv2
        keyframes = []
        interval = max(1, frame_count // num_keyframes)

        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                keyframes.append(i)

        return keyframes

    def _analyze_consistency(self, cap, frame_count: int) -> float:
        """分析视频内容一致性"""
        try:
            import cv2
            # 采样几帧计算相似度
            samples = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
            frames = []

            for pos in samples:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray)

            # 计算帧之间的相似度
            if len(frames) < 2:
                return 0.5

            similarities = []
            for i in range(len(frames) - 1):
                diff = cv2.absdiff(frames[i], frames[i + 1])
                similarity = 1.0 - (np.mean(diff) / 255.0)
                similarities.append(similarity)

            return np.mean(similarities) if similarities else 0.5
        except:
            return 0.5

class AudioAnalyzer:
    """音频分析 - 音频质量、噪声检测"""
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            import librosa

            # 加载音频
            y, sr = librosa.load(file_path, sr=None)

            # 音频时长
            duration = librosa.get_duration(y=y, sr=sr)

            # 音频质量评分（基于采样率）
            quality_score = min(sr / 44100, 1.0)

            # 噪声检测（基于频谱熵）
            noise_score = self._detect_noise(y, sr)

            overall = (quality_score * 0.6 + (1 - noise_score) * 0.4) * 100

            return {
                "duration": round(duration, 2),
                "sample_rate": sr,
                "quality": round(quality_score, 3),
                "noise": round(noise_score, 3),
                "score": round(overall, 2)
            }
        except Exception as e:
            return {"error": f"音频分析失败: {str(e)}", "score": 0}

    def _detect_noise(self, y, sr: int) -> float:
        """检测音频中的噪声"""
        try:
            import librosa
            # 计算频谱质心
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_db = librosa.power_to_db(S, ref=np.max)

            # 计算频谱熵作为噪声指标
            entropy = -np.sum(S_db * np.log2(S_db + 1e-10)) / S_db.size

            # 归一化到0-1
            noise_score = min(entropy / 10, 1.0)
            return noise_score
        except:
            return 0.2

class EvidenceFusion:
    """证据融合 - 动态权重调整"""

    @staticmethod
    def fuse(text_score: float = None, image_score: float = None,
             video_score: float = None, audio_score: float = None,
             title_consistency_score: float = None,
             image_text_consistency_score: float = None) -> Dict[str, Any]:
        """
        动态加权融合多模态分析结果
        根据用户提交的模态类型动态调整权重
        """
        # 收集有效的模态分数
        modalities = {}
        if text_score is not None and text_score > 0:
            modalities['text'] = text_score
        if image_score is not None and image_score > 0:
            modalities['image'] = image_score
        if video_score is not None and video_score > 0:
            modalities['video'] = video_score
        if audio_score is not None and audio_score > 0:
            modalities['audio'] = audio_score

        # 如果没有任何有效模态，返回0
        if not modalities:
            return {
                "overall_score": 0,
                "credibility_label": "无法判定",
                "credibility_description": "未提供有效的分析内容",
                "modalities_used": [],
                "weights": {},
                "details": {}
            }

        # 动态计算权重
        weights = EvidenceFusion._calculate_dynamic_weights(list(modalities.keys()))

        # 计算加权分数
        weighted_sum = 0
        for modality, score in modalities.items():
            weighted_sum += score * weights[modality]

        # 一致性惩罚（如果存在不一致，降低总分）
        consistency_penalty = 0
        consistency_issues = []

        if title_consistency_score is not None and title_consistency_score < 0.4:
            penalty = (0.4 - title_consistency_score) * 15  # 最多扣15分
            consistency_penalty += penalty
            consistency_issues.append(f"标题-正文不一致(扣{penalty:.1f}分)")

        if image_text_consistency_score is not None and image_text_consistency_score < 0.3:
            penalty = (0.3 - image_text_consistency_score) * 15  # 最多扣15分
            consistency_penalty += penalty
            consistency_issues.append(f"图片-文本不一致(扣{penalty:.1f}分)")

        # 最终分数
        final_score = max(0, weighted_sum - consistency_penalty)

        # 获取可信度标签和描述
        credibility_label = get_credibility_label(final_score)
        credibility_description = get_credibility_description(final_score)

        return {
            "overall_score": round(final_score, 2),
            "credibility_label": credibility_label,
            "credibility_description": credibility_description,
            "modalities_used": list(modalities.keys()),
            "weights": {k: round(v, 2) for k, v in weights.items()},
            "modality_scores": {k: round(v, 2) for k, v in modalities.items()},
            "consistency_penalty": round(consistency_penalty, 2),
            "consistency_issues": consistency_issues,
            "thresholds": CREDIBILITY_THRESHOLDS
        }

    @staticmethod
    def _calculate_dynamic_weights(modalities: List[str]) -> Dict[str, float]:
        """
        根据提交的模态类型动态计算权重
        原则：
        - 文本是最重要的模态，权重最高
        - 有多个模态时，分散权重
        - 单一模态时，该模态权重为100%
        """
        # 基础权重配置
        base_weights = {
            'text': 0.45,   # 文本基础权重最高
            'image': 0.25,  # 图像次之
            'video': 0.20,  # 视频
            'audio': 0.10   # 音频
        }

        # 只有一个模态时，权重为100%
        if len(modalities) == 1:
            return {modalities[0]: 1.0}

        # 多个模态时，按比例分配
        total_base = sum(base_weights[m] for m in modalities)
        weights = {}
        for m in modalities:
            weights[m] = base_weights[m] / total_base

        return weights

    @staticmethod
    def fuse_simple(text_score: float, image_score: float, video_score: float, audio_score: float) -> float:
        """简单融合（向后兼容）"""
        result = EvidenceFusion.fuse(text_score, image_score, video_score, audio_score)
        return result["overall_score"]

# 创建分析器实例
text_analyzer = TextAnalyzer()
image_analyzer = ImageAnalyzer()
video_analyzer = VideoAnalyzer()
audio_analyzer = AudioAnalyzer()

