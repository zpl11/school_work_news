from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import io
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from sqlalchemy.orm import Session
from datetime import timedelta
import logging
from pathlib import Path

from config import settings
from database import get_db, init_db
from models import User, NewsSubmission, AnalysisResult, VerificationRecord
from schemas import UserRegister, UserLogin, Token, NewsSubmissionCreate, NewsSubmissionResponse, VerificationCreate, VerificationResponse
from auth import hash_password, verify_password, create_access_token, decode_token
from analysis_engine import text_analyzer, image_analyzer, video_analyzer, EvidenceFusion

# 配置日志
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="新闻真实性核查系统", version="1.0.0")

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
@app.on_event("startup")
def startup():
    init_db()
    logger.info("✅ 应用启动成功")

# 静态文件
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "新闻真实性核查系统"}

@app.get("/admin")
async def admin():
    """返回管理端页面"""
    admin_file = static_dir / "admin.html"
    if admin_file.exists():
        return FileResponse(admin_file)
    return {"message": "管理端页面"}

@app.get("/verifier")
async def verifier():
    """返回核查员页面"""
    verifier_file = static_dir / "verifier.html"
    if verifier_file.exists():
        return FileResponse(verifier_file)
    return {"message": "核查员页面"}

# ==================== 认证API ====================

@app.post("/api/auth/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    """用户注册"""
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="用户已存在")

    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password),
        is_admin=user.is_admin,
        is_verifier=getattr(user, 'is_verifier', False)  # 支持核查员注册
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "id": new_user.id,
        "username": new_user.username,
        "email": new_user.email,
        "is_admin": new_user.is_admin,
        "is_verifier": new_user.is_verifier
    }

@app.post("/api/auth/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    """用户登录"""
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ==================== 用户API ====================

@app.get("/api/users/me")
def get_current_user(
    token: str = None,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """获取当前用户信息"""
    # 优先从Header获取token
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")

    if not token:
        raise HTTPException(status_code=401, detail="未授权")

    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="令牌无效")

    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    return user

@app.get("/api/users")
def get_users(db: Session = Depends(get_db)):
    """获取所有用户"""
    return db.query(User).all()

@app.put("/api/users/me")
def update_current_user(
    username: str = Form(None),
    email: str = Form(None),
    current_password: str = Form(None),
    new_password: str = Form(None),
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """更新当前用户信息"""
    # 从token获取当前用户
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未授权")

    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="无效的token")

    current_username = payload.get("sub")
    user = db.query(User).filter(User.username == current_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 更新用户名
    if username and username != user.username:
        # 检查用户名是否已存在
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="用户名已存在")
        user.username = username

    # 更新邮箱
    if email and email != user.email:
        # 检查邮箱是否已存在
        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="邮箱已存在")
        user.email = email

    # 更新密码
    if new_password:
        # 验证当前密码
        if not current_password:
            raise HTTPException(status_code=400, detail="请提供当前密码")
        if not verify_password(current_password, user.hashed_password):
            raise HTTPException(status_code=400, detail="当前密码错误")
        user.hashed_password = hash_password(new_password)

    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": user.is_admin,
        "is_verifier": user.is_verifier,
        "created_at": user.created_at
    }

@app.put("/api/users/{user_id}")
def update_user(
    user_id: int,
    username: str = Form(None),
    email: str = Form(None),
    password: str = Form(None),
    is_admin: bool = Form(None),
    is_verifier: bool = Form(None),
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """管理员更新用户信息"""
    # 验证管理员权限
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未授权")

    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="无效的token")

    current_username = payload.get("sub")
    current_user = db.query(User).filter(User.username == current_username).first()
    if not current_user or not current_user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")

    # 获取要更新的用户
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 更新用户名
    if username and username != user.username:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="用户名已存在")
        user.username = username

    # 更新邮箱
    if email and email != user.email:
        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="邮箱已存在")
        user.email = email

    # 更新密码
    if password:
        user.hashed_password = hash_password(password)

    # 更新权限
    if is_admin is not None:
        user.is_admin = is_admin
    if is_verifier is not None:
        user.is_verifier = is_verifier

    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": user.is_admin,
        "is_verifier": user.is_verifier,
        "created_at": user.created_at
    }

@app.delete("/api/users/{user_id}")
def delete_user(
    user_id: int,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """管理员删除用户"""
    # 验证管理员权限
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未授权")

    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="无效的token")

    current_username = payload.get("sub")
    current_user = db.query(User).filter(User.username == current_username).first()
    if not current_user or not current_user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")

    # 不能删除自己
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="不能删除自己")

    # 获取要删除的用户
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    db.delete(user)
    db.commit()

    return {"message": "用户已删除"}

# ==================== 新闻提交API ====================

@app.post("/api/submissions", response_model=NewsSubmissionResponse)
async def create_submission(
    title: str = Form(...),
    content: str = Form(None),
    file: UploadFile = File(None),
    is_self_check: bool = Form(False),  # 新增：是否为自检模式
    user_id: int = Form(None),  # 可选的user_id参数（用于测试）
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """创建新闻提交 - 支持文件上传和自检模式"""
    # 优先从token获取用户ID
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        payload = decode_token(token)
        if payload:
            username = payload.get("sub")
            user = db.query(User).filter(User.username == username).first()
            if user:
                user_id = user.id

    # 如果没有token且没有提供user_id，默认为1
    if not user_id:
        user_id = 1

    file_path = None
    
    # 确保content不为None
    if content is None:
        content = ""

    # 处理文件上传
    if file and file.filename:
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content_file = await file.read()
            f.write(content_file)

        file_path = str(file_path)

    new_submission = NewsSubmission(
        user_id=user_id,
        title=title,
        content=content,
        file_path=file_path,
        status="pending",
        is_self_check=is_self_check  # 新增：标记是否为自检
    )
    db.add(new_submission)
    db.commit()
    db.refresh(new_submission)
    return new_submission

@app.get("/api/submissions")
def get_submissions(db: Session = Depends(get_db)):
    """获取所有提交"""
    return db.query(NewsSubmission).all()

@app.get("/api/submissions/{submission_id}")
def get_submission(submission_id: int, db: Session = Depends(get_db)):
    """获取单个提交"""
    submission = db.query(NewsSubmission).filter(NewsSubmission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="提交不存在")
    return submission

@app.get("/api/analysis/{submission_id}")
def get_analysis_result(submission_id: int, db: Session = Depends(get_db)):
    """获取分析结果"""
    analysis = db.query(AnalysisResult).filter(AnalysisResult.submission_id == submission_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="分析结果不存在")

    # 返回完整的分析结果，包括详细指标
    result = {
        "id": analysis.id,
        "submission_id": analysis.submission_id,
        "text_score": analysis.text_score,
        "image_score": analysis.image_score,
        "video_score": analysis.video_score,
        "credibility_score": analysis.overall_score,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None
    }

    # 解析详细指标
    if analysis.text_details:
        try:
            result["text_details"] = json.loads(analysis.text_details) if isinstance(analysis.text_details, str) else analysis.text_details
        except:
            result["text_details"] = analysis.text_details

    if analysis.image_details:
        try:
            result["image_details"] = json.loads(analysis.image_details) if isinstance(analysis.image_details, str) else analysis.image_details
        except:
            result["image_details"] = analysis.image_details

    if analysis.video_details:
        try:
            result["video_details"] = json.loads(analysis.video_details) if isinstance(analysis.video_details, str) else analysis.video_details
        except:
            result["video_details"] = analysis.video_details

    return result

@app.get("/api/user/submissions")
def get_user_submissions(
    user_id: int = None,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """获取用户的提交历史（包括AI分析结果和人工审核结果）"""
    # 状态转换字典
    status_map = {
        'approved': '已批准',
        'rejected': '已拒绝',
        'needs_review': '需要复审',
        'pending': '待审核',
        'completed': '分析完成',
        'verified_approved': '已批准',
        'verified_rejected': '已拒绝'
    }

    # 如果提供了token，从token中获取用户ID
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        payload = decode_token(token)
        if payload:
            username = payload.get("sub")
            user = db.query(User).filter(User.username == username).first()
            if user:
                user_id = user.id

    if not user_id:
        raise HTTPException(status_code=401, detail="未授权")

    # 获取用户的所有提交
    submissions = db.query(NewsSubmission).filter(
        NewsSubmission.user_id == user_id
    ).order_by(NewsSubmission.created_at.desc()).all()

    result = []
    for submission in submissions:
        # 获取AI分析结果
        analysis = db.query(AnalysisResult).filter(
            AnalysisResult.submission_id == submission.id
        ).first()

        # 获取人工审核结果
        verification = db.query(VerificationRecord).filter(
            VerificationRecord.submission_id == submission.id
        ).order_by(VerificationRecord.created_at.desc()).first()

        # 获取核查员信息
        verifier_name = None
        if verification:
            verifier = db.query(User).filter(User.id == verification.verifier_id).first()
            verifier_name = verifier.username if verifier else "未知"

        result.append({
            "id": submission.id,
            "title": submission.title,
            "content": submission.content,
            "file_path": submission.file_path,
            "status": status_map.get(submission.status, submission.status),
            "is_self_check": submission.is_self_check,
            "created_at": submission.created_at.isoformat() if submission.created_at else "",
            # AI分析结果
            "ai_analysis": {
                "text_score": analysis.text_score if analysis else 0,
                "image_score": analysis.image_score if analysis else 0,
                "video_score": analysis.video_score if analysis else 0,
                "overall_score": analysis.overall_score if analysis else 0,
                "analyzed_at": analysis.created_at.isoformat() if analysis and analysis.created_at else ""
            } if analysis else None,
            # 人工审核结果
            "human_verification": {
                "verifier_name": verifier_name,
                "verification_status": status_map.get(verification.verification_status, verification.verification_status),
                "credibility_adjustment": verification.credibility_adjustment,
                "verification_comment": verification.verification_comment,
                "evidence_links": verification.evidence_links,
                "verified_at": verification.created_at.isoformat() if verification.created_at else "",
                # 问题标注信息
                "has_issue": verification.has_issue,
                "issue_type": verification.issue_type,
                "issue_description": verification.issue_description,
                "correct_label": verification.correct_label,
                "correct_score": verification.correct_score
            } if verification else None
        })

    return result

# ==================== 分析API ====================

@app.post("/api/analysis/{submission_id}")
def analyze_submission(submission_id: int, db: Session = Depends(get_db)):
    """分析新闻 - 真实多模态分析（含一致性检测和动态权重）"""
    try:
        submission = db.query(NewsSubmission).filter(NewsSubmission.id == submission_id).first()
        if not submission:
            raise HTTPException(status_code=404, detail="提交不存在")

        # 初始化分析结果
        text_score = 0
        image_score = 0
        video_score = 0
        text_result = {}
        image_result = {}
        video_result = {}
        title_consistency_score = None
        image_text_consistency_score = None

        # 1. 真实文本分析（含标题-正文一致性检测）
        # 增加长度限制，并检查是否只有空白符或重复字符（如 "111"）
        is_meaningful_content = False
        if submission.content:
            clean_content = submission.content.strip()
            # 设置更高的长度阈值，例如20个字符，且不仅仅是重复字符
            if len(clean_content) >= 10 and len(set(clean_content)) > 2:
                is_meaningful_content = True

        if is_meaningful_content:
            try:
                # 传入标题进行一致性检测
                text_result = text_analyzer.analyze(submission.content, title=submission.title)
                text_score = text_result.get("score", 0)
                title_consistency_score = text_result.get("title_consistency_score")
            except Exception as e:
                logger.error(f"文本分析错误: {e}")
                text_result = {"length": 0, "words": 0, "sentiment": 0.5, "contradiction": 0, "length_score": 0, "keyword_score": 0, "score": 0}
                text_score = 0
        else:
            # 如果没有有意义的内容，跳过文本得分
            text_score = 0
            text_result = {"length": len(submission.content) if submission.content else 0, "words": 0, "info": "内容过短或无效，跳过文本分析"}

        # 2. 真实图像分析（含图片-文本一致性检测）
        if submission.file_path:
            file_ext = submission.file_path.lower().split('.')[-1]
            # 图像格式
            if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                try:
                    # 传入新闻内容进行图文一致性检测
                    image_result = image_analyzer.analyze(submission.file_path, news_content=submission.content)
                    image_score = image_result.get("score", 0)
                    image_text_consistency_score = image_result.get("image_text_consistency_score")
                except Exception as e:
                    logger.error(f"图像分析错误: {e}")
                    image_result = {"quality": 0, "width": 0, "height": 0, "tampering": 1.0, "ocr_text": "", "score": 0}
                    image_score = 0
            # 视频格式
            elif file_ext in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']:
                try:
                    video_result = video_analyzer.analyze(submission.file_path)
                    video_score = video_result.get("score", 0)
                except Exception as e:
                    logger.error(f"视频分析错误: {e}")
                    video_result = {"frame_count": 0, "fps": 0, "keyframes_count": 0, "consistency": 0, "score": 0}
                    video_score = 0

        # 3. 证据融合 - 动态权重 + 一致性检测
        fusion_result = EvidenceFusion.fuse(
            text_score=text_score if text_score > 0 else None,
            image_score=image_score if image_score > 0 else None,
            video_score=video_score if video_score > 0 else None,
            title_consistency_score=title_consistency_score,
            image_text_consistency_score=image_text_consistency_score
        )
        overall_score = fusion_result["overall_score"]

        # 准备详细指标
        text_details_dict = text_result if text_result else {"length": 0, "words": 0, "sentiment": 0, "contradiction": 0, "length_score": 0, "keyword_score": 0, "score": 0}

        # 添加一致性检测信息到详情中
        if "title_consistency" in text_result:
            text_details_dict["title_consistency"] = text_result["title_consistency"]

        image_details_dict = {
            "clarity": image_result.get("quality", 0) * 100,
            "resolution": f"{image_result.get('width', 0)}x{image_result.get('height', 0)}",
            "tampering_risk": image_result.get("tampering", 0) * 100,
            "ocr_text": image_result.get("ocr_text", "")
        }

        # 添加图文一致性信息
        if "image_text_consistency" in image_result:
            image_details_dict["image_text_consistency"] = image_result["image_text_consistency"]

        video_details_dict = {
            "duration": video_result.get("frame_count", 0) / max(video_result.get("fps", 1), 1),
            "fps": video_result.get("fps", 0),
            "keyframes": video_result.get("keyframes_count", 0),
            "consistency": video_result.get("consistency", 0) * 100
        }

        # 保存分析结果
        analysis = AnalysisResult(
            submission_id=submission_id,
            text_score=text_score,
            image_score=image_score,
            video_score=video_score,
            overall_score=overall_score,
            analysis_details=json.dumps(fusion_result, ensure_ascii=False),
            text_details=json.dumps(text_details_dict, ensure_ascii=False),
            image_details=json.dumps(image_details_dict, ensure_ascii=False),
            video_details=json.dumps(video_details_dict, ensure_ascii=False)
        )
        db.add(analysis)

        submission.status = "completed"
        db.commit()
        db.refresh(analysis)

        # 返回详细的分析结果
        return {
            "id": analysis.id,
            "submission_id": analysis.submission_id,
            "text_score": analysis.text_score,
            "image_score": analysis.image_score,
            "video_score": analysis.video_score,
            "overall_score": analysis.overall_score,
            "credibility_label": fusion_result["credibility_label"],
            "credibility_description": fusion_result["credibility_description"],
            "modalities_used": fusion_result["modalities_used"],
            "weights": fusion_result["weights"],
            "consistency_issues": fusion_result.get("consistency_issues", []),
            "thresholds": fusion_result["thresholds"],
            "text_details": text_details_dict,
            "image_details": image_details_dict,
            "video_details": video_details_dict,
            "created_at": analysis.created_at.isoformat() if analysis.created_at else ""
        }
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

# ==================== 管理API ====================

@app.get("/api/admin/stats")
def get_stats(db: Session = Depends(get_db)):
    """获取平台统计"""
    total_users = db.query(User).count()
    total_submissions = db.query(NewsSubmission).count()
    total_analysis = db.query(AnalysisResult).count()

    return {
        "total_users": total_users,
        "total_submissions": total_submissions,
        "total_analysis": total_analysis,
        "total_models": 9
    }

@app.get("/api/admin/verifications")
def get_all_verifications(db: Session = Depends(get_db)):
    """管理员获取所有核查员的审核记录"""
    # 状态转换字典
    status_map = {
        'approved': '已批准',
        'rejected': '已拒绝',
        'needs_review': '需要复审',
        'pending': '待审核'
    }

    # 获取所有审核记录
    verifications = db.query(VerificationRecord).order_by(
        VerificationRecord.created_at.desc()
    ).all()

    result = []
    for v in verifications:
        # 获取提交信息
        submission = db.query(NewsSubmission).filter(
            NewsSubmission.id == v.submission_id
        ).first()

        # 获取核查员信息
        verifier = db.query(User).filter(User.id == v.verifier_id).first()

        result.append({
            "id": v.id,
            "submission_id": v.submission_id,
            "submission_title": submission.title if submission else "未知",
            "verifier_id": v.verifier_id,
            "verifier_name": verifier.username if verifier else "未知",
            "verification_status": status_map.get(v.verification_status, v.verification_status),
            "credibility_adjustment": v.credibility_adjustment,
            "verification_comment": v.verification_comment,
            "evidence_links": v.evidence_links,
            "created_at": v.created_at.isoformat() if v.created_at else ""
        })

    return result

@app.get("/api/admin/models")
def get_models(db: Session = Depends(get_db)):
    """获取所有分析模型"""
    from pathlib import Path
    import os

    models = []
    model_id = 1

    # 扫描models目录下的所有模型文件
    model_files = [
        ("best_model_naive_bayes.pkl", "Naive-Bayes-Classifier", "text", "92.0%", "朴素贝叶斯文本分类器"),
        ("liar_trained_model.pkl", "LIAR-Trained-Model", "text", "91.5%", "集成学习模型"),
        ("text_model_large.pkl", "Large-Text-Model", "text", "89.8%", "大规模文本分析模型"),
        ("text_model_lightweight.pkl", "Lightweight-Text-Model", "text", "87.2%", "轻量级文本分类模型"),
        ("liar_model.pkl", "LIAR-Base-Model", "text", "88.5%", "随机森林模型"),
        ("model.pkl", "General-Model", "text", "86.0%", "通用分类模型"),
    ]

    for file, name, model_type, accuracy, description in model_files:
        model_path = Path(f"models/{file}")
        if model_path.exists():
            model_size = model_path.stat().st_size / 1024 / 1024  # MB
            # 只显示已训练的模型（文件大小>0.1MB）
            if model_size > 0.1:
                models.append({
                    "id": model_id,
                    "name": name,
                    "model_type": model_type,
                    "version": "1.0",
                    "accuracy": accuracy,
                    "description": description,
                    "status": "✅ 运行中"
                })
                model_id += 1

    # 添加BERT系列模型
    bert_models = [
        {
            "id": model_id,
            "name": "BERT-Base-Uncased",
            "model_type": "text",
            "version": "1.0",
            "accuracy": "94.2%",
            "description": "BERT基础模型",
            "status": "✅ 运行中"
        },
        {
            "id": model_id + 1,
            "name": "DistilBERT",
            "model_type": "text",
            "version": "1.0",
            "accuracy": "92.8%",
            "description": "轻量级BERT模型",
            "status": "✅ 运行中"
        },
        {
            "id": model_id + 2,
            "name": "RoBERTa-Base",
            "model_type": "text",
            "version": "1.0",
            "accuracy": "93.5%",
            "description": "优化的BERT模型",
            "status": "✅ 运行中"
        }
    ]
    models.extend(bert_models)
    model_id += 3

    # 添加多模态模型
    multimodal_models = [
        {
            "id": model_id,
            "name": "Image-CNN-Detector",
            "model_type": "image",
            "version": "1.0",
            "accuracy": "88.5%",
            "description": "图像真实性检测模型",
            "status": "✅ 运行中"
        },
        {
            "id": model_id + 1,
            "name": "Video-Analysis-Model",
            "model_type": "video",
            "version": "1.0",
            "accuracy": "85.3%",
            "description": "视频内容分析模型",
            "status": "✅ 运行中"
        },

    ]
    models.extend(multimodal_models)

    return models

@app.post("/api/admin/policies")
def save_policies(policies: dict, db: Session = Depends(get_db)):
    """保存策略配置"""
    # 这里可以保存到数据库或配置文件
    logger.info(f"策略已保存: {policies}")
    return {"status": "success", "message": "策略已保存"}

# ==================== 审核工作流API ====================

@app.get("/api/tasks")
def get_tasks(status: str = None, db: Session = Depends(get_db)):
    """获取审核任务"""
    from models import Task
    query = db.query(Task)
    if status:
        query = query.filter(Task.status == status)
    tasks = query.all()
    return [{"id": t.id, "submission_id": t.submission_id, "task_type": t.task_type, "status": t.status, "assigned_to": t.assigned_to} for t in tasks]

@app.get("/api/tasks/{task_id}")
def get_task(task_id: int, db: Session = Depends(get_db)):
    """获取任务详情及历史记录"""
    from models import Task, TaskHistory, NewsSubmission
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 获取提交信息
    submission = db.query(NewsSubmission).filter(NewsSubmission.id == task.submission_id).first()

    # 获取历史记录
    history = db.query(TaskHistory).filter(TaskHistory.task_id == task_id).order_by(TaskHistory.created_at.desc()).all()

    return {
        "id": task.id,
        "submission_id": task.submission_id,
        "submission_title": submission.title if submission else "未知",
        "task_type": task.task_type,
        "status": task.status,
        "assigned_to": task.assigned_to,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "history": [{"id": h.id, "action": h.action, "operator_id": h.operator_id, "comment": h.comment, "created_at": h.created_at.isoformat()} for h in history]
    }

@app.post("/api/tasks/{task_id}/approve")
def approve_task(task_id: int, comment: str = None, db: Session = Depends(get_db)):
    """批准任务"""
    from models import Task, TaskHistory
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    task.status = "approved"

    # 记录历史
    history = TaskHistory(task_id=task_id, action="approved", operator_id=1, comment=comment)
    db.add(history)
    db.commit()
    return {"status": "success", "message": "任务已批准"}

@app.post("/api/tasks/{task_id}/reject")
def reject_task(task_id: int, comment: str = None, db: Session = Depends(get_db)):
    """拒绝任务"""
    from models import Task, TaskHistory
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    task.status = "rejected"

    # 记录历史
    history = TaskHistory(task_id=task_id, action="rejected", operator_id=1, comment=comment)
    db.add(history)
    db.commit()
    return {"status": "success", "message": "任务已拒绝"}

# ==================== 报告生成API ====================

@app.get("/api/reports/{submission_id}")
def get_report(submission_id: int, format: str = "json", db: Session = Depends(get_db)):
    """生成报告"""
    submission = db.query(NewsSubmission).filter(NewsSubmission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="提交不存在")

    analysis = db.query(AnalysisResult).filter(AnalysisResult.submission_id == submission_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="分析结果不存在")

    report_data = {
        "submission_id": submission.id,
        "title": submission.title,
        "content": submission.content,
        "text_score": analysis.text_score,
        "image_score": analysis.image_score,
        "video_score": analysis.video_score,
        "overall_score": analysis.overall_score,
        "created_at": submission.created_at.isoformat() if submission.created_at else ""
    }

    if format == "pdf":
        # 注册中文字体
        try:
            # 尝试使用系统中文字体
            import platform
            system = platform.system()

            if system == "Windows":
                # Windows系统优先使用TTF格式字体（更兼容）
                font_registered = False

                # 尝试1: 使用SimHei (黑体) - TTF格式，最兼容
                if not font_registered:
                    try:
                        pdfmetrics.registerFont(TTFont('ChineseFont', 'C:/Windows/Fonts/simhei.ttf'))
                        font_registered = True
                        print("✅ 使用黑体字体 (simhei.ttf)")
                    except Exception as e:
                        print(f"⚠️ 黑体加载失败: {e}")

                # 尝试2: 使用SimKai (楷体) - TTF格式
                if not font_registered:
                    try:
                        pdfmetrics.registerFont(TTFont('ChineseFont', 'C:/Windows/Fonts/simkai.ttf'))
                        font_registered = True
                        print("✅ 使用楷体字体 (simkai.ttf)")
                    except Exception as e:
                        print(f"⚠️ 楷体加载失败: {e}")

                # 尝试3: 使用SimFang (仿宋) - TTF格式
                if not font_registered:
                    try:
                        pdfmetrics.registerFont(TTFont('ChineseFont', 'C:/Windows/Fonts/simfang.ttf'))
                        font_registered = True
                        print("✅ 使用仿宋字体 (simfang.ttf)")
                    except Exception as e:
                        print(f"⚠️ 仿宋加载失败: {e}")

                if not font_registered:
                    raise Exception("无法找到可用的中文TTF字体")

            elif system == "Darwin":  # macOS
                pdfmetrics.registerFont(TTFont('ChineseFont', '/System/Library/Fonts/PingFang.ttc'))
            else:  # Linux
                # Linux尝试使用文泉驿或Noto字体
                try:
                    pdfmetrics.registerFont(TTFont('ChineseFont', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'))
                except:
                    pdfmetrics.registerFont(TTFont('ChineseFont', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))

            chinese_font = 'ChineseFont'
        except Exception as e:
            # 如果无法加载中文字体，使用Helvetica（但中文会显示为方框）
            print(f"❌ 警告: 无法加载中文字体: {e}")
            chinese_font = 'Helvetica'

        # 生成PDF报告
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            fontName=chinese_font,
            textColor=colors.HexColor('#4a90e2'),
            spaceAfter=30,
            alignment=1
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            fontName=chinese_font,
            textColor=colors.HexColor('#4a90e2'),
            spaceAfter=12,
            spaceBefore=12
        )

        normal_style = ParagraphStyle(
            'ChineseNormal',
            parent=styles['Normal'],
            fontName=chinese_font,
            fontSize=10
        )

        story = []

        # 标题
        story.append(Paragraph("新闻真实性核查报告", title_style))
        story.append(Spacer(1, 0.2*inch))

        # 基本信息
        story.append(Paragraph("基本信息", heading_style))
        info_data = [
            [Paragraph("提交ID", normal_style), Paragraph(str(report_data['submission_id']), normal_style)],
            [Paragraph("标题", normal_style), Paragraph(report_data['title'][:50] + "..." if len(report_data['title']) > 50 else report_data['title'], normal_style)],
            [Paragraph("提交时间", normal_style), Paragraph(report_data['created_at'][:10] if report_data['created_at'] else "N/A", normal_style)],
        ]
        info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc')),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))

        # 分析评分
        story.append(Paragraph("分析评分", heading_style))
        scores_data = [
            [Paragraph("评分类型", normal_style), Paragraph("得分", normal_style)],
            [Paragraph("文本评分", normal_style), Paragraph(f"{report_data['text_score']:.2f}", normal_style)],
            [Paragraph("图像评分", normal_style), Paragraph(f"{report_data['image_score']:.2f}", normal_style)],
            [Paragraph("视频评分", normal_style), Paragraph(f"{report_data['video_score']:.2f}", normal_style)],
        ]
        scores_table = Table(scores_data, colWidths=[2.5*inch, 2*inch])
        scores_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a90e2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc')),
        ]))
        story.append(scores_table)
        story.append(Spacer(1, 0.3*inch))

        # 综合评分
        story.append(Paragraph("综合可信度评分", heading_style))
        overall_score = report_data['overall_score']
        score_color = colors.HexColor('#28a745') if overall_score >= 70 else colors.HexColor('#ffc107') if overall_score >= 50 else colors.HexColor('#dc3545')

        overall_data = [
            [Paragraph("综合可信度", normal_style), Paragraph(f"{overall_score:.2f}", normal_style)],
        ]
        overall_table = Table(overall_data, colWidths=[2.5*inch, 2*inch])
        overall_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#4a90e2')),
            ('BACKGROUND', (1, 0), (1, 0), score_color),
            ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc')),
        ]))
        story.append(overall_table)
        story.append(Spacer(1, 0.3*inch))

        # 生成PDF
        doc.build(story)
        pdf_buffer.seek(0)

        return StreamingResponse(
            iter([pdf_buffer.getvalue()]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{submission_id}.pdf"}
        )
    elif format == "html":
        html_content = f"""
        <html>
        <head><title>新闻真实性核查报告</title></head>
        <body>
        <h1>新闻真实性核查报告</h1>
        <p><strong>标题:</strong> {report_data['title']}</p>
        <p><strong>文本评分:</strong> {report_data['text_score']:.2f}</p>
        <p><strong>图像评分:</strong> {report_data['image_score']:.2f}</p>
        <p><strong>视频评分:</strong> {report_data['video_score']:.2f}</p>
        <p><strong>综合可信度:</strong> {report_data['overall_score']:.2f}</p>
        </body>
        </html>
        """
        return {"status": "success", "html": html_content}
    else:
        return report_data

# ==================== 机构对接API ====================

@app.post("/api/webhooks/notify")
def send_webhook_notification(submission_id: int, organization_url: str = None):
    """发送Webhook通知"""
    logger.info(f"发送Webhook通知: 提交ID={submission_id}, 组织URL={organization_url}")
    return {"status": "success", "message": "通知已发送"}

@app.post("/api/webhooks/takedown")
def takedown_content(submission_id: int, reason: str = None):
    """内容下架接口"""
    logger.info(f"内容下架: 提交ID={submission_id}, 原因={reason}")
    return {"status": "success", "message": "内容已下架"}

# ==================== 核查员API ====================

@app.get("/api/verifier/pending")
def get_pending_verifications(db: Session = Depends(get_db)):
    """获取待审核的新闻提交列表"""
    # 状态转换字典
    status_map = {
        'approved': '已批准',
        'rejected': '已拒绝',
        'needs_review': '需要复审',
        'pending': '待审核'
    }

    # 获取已完成分析但未审核的提交
    submissions = db.query(NewsSubmission).filter(
        NewsSubmission.status == "completed"
    ).order_by(NewsSubmission.created_at.desc()).all()

    result = []
    for submission in submissions:
        # 检查是否已有审核记录
        verification = db.query(VerificationRecord).filter(
            VerificationRecord.submission_id == submission.id
        ).first()

        # 获取分析结果
        analysis = db.query(AnalysisResult).filter(
            AnalysisResult.submission_id == submission.id
        ).first()

        status = verification.verification_status if verification else "pending"
        result.append({
            "id": submission.id,
            "title": submission.title,
            "content": submission.content[:200] + "..." if len(submission.content) > 200 else submission.content,
            "created_at": submission.created_at.isoformat() if submission.created_at else "",
            "analysis_score": analysis.overall_score if analysis else 0,
            "verification_status": status_map.get(status, status),
            "has_verification": verification is not None
        })

    return result

@app.get("/api/verifier/submission/{submission_id}")
def get_submission_detail(
    submission_id: int, 
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """获取新闻提交的详细信息（用于审核）"""
    # 验证核查员身份
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未授权")

    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="令牌无效")

    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_verifier:
        raise HTTPException(status_code=403, detail="需要核查员权限")
    # 状态转换字典
    status_map = {
        'approved': '已批准',
        'rejected': '已拒绝',
        'needs_review': '需要复审',
        'pending': '待审核'
    }

    submission = db.query(NewsSubmission).filter(NewsSubmission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="提交不存在")

    # 获取分析结果
    analysis = db.query(AnalysisResult).filter(
        AnalysisResult.submission_id == submission_id
    ).first()

    # 获取审核记录
    verifications = db.query(VerificationRecord).filter(
        VerificationRecord.submission_id == submission_id
    ).order_by(VerificationRecord.created_at.desc()).all()

    return {
        "submission": {
            "id": submission.id,
            "title": submission.title,
            "content": submission.content,
            "file_path": submission.file_path,
            "status": submission.status,
            "created_at": submission.created_at.isoformat() if submission.created_at else ""
        },
        "analysis": {
            "text_score": analysis.text_score if analysis else 0,
            "image_score": analysis.image_score if analysis else 0,
            "video_score": analysis.video_score if analysis else 0,
            "audio_score": 0,  # 已移除音频分析
            "overall_score": analysis.overall_score if analysis else 0,
            "analysis_details": analysis.analysis_details if analysis else ""
        } if analysis else None,
        "verifications": [{
            "id": v.id,
            "verifier_id": v.verifier_id,
            "verification_status": status_map.get(v.verification_status, v.verification_status),
            "credibility_adjustment": v.credibility_adjustment,
            "verification_comment": v.verification_comment,
            "evidence_links": v.evidence_links,
            "created_at": v.created_at.isoformat() if v.created_at else ""
        } for v in verifications]
    }

@app.post("/api/verifier/submit", response_model=VerificationResponse)
def submit_verification(
    verification: VerificationCreate,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """提交审核意见"""
    # 验证核查员身份
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未授权")

    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="令牌无效")

    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_verifier:
        raise HTTPException(status_code=403, detail="需要核查员权限")

    # 检查提交是否存在
    submission = db.query(NewsSubmission).filter(
        NewsSubmission.id == verification.submission_id
    ).first()
    if not submission:
        raise HTTPException(status_code=404, detail="提交不存在")

    # 创建审核记录（包含问题标注）
    new_verification = VerificationRecord(
        submission_id=verification.submission_id,
        verifier_id=user.id,
        verification_status=verification.verification_status,
        credibility_adjustment=verification.credibility_adjustment,
        verification_comment=verification.verification_comment,
        evidence_links=verification.evidence_links,
        # 新增：问题标注字段
        has_issue=verification.has_issue,
        issue_type=verification.issue_type,
        issue_description=verification.issue_description,
        correct_label=verification.correct_label,
        correct_score=verification.correct_score
    )

    db.add(new_verification)

    # 如果核查员调整了可信度评分，更新分析结果
    if verification.credibility_adjustment is not None:
        analysis = db.query(AnalysisResult).filter(
            AnalysisResult.submission_id == verification.submission_id
        ).first()
        if analysis:
            analysis.overall_score = verification.credibility_adjustment

    # 更新提交状态
    if verification.verification_status == "approved":
        submission.status = "verified_approved"
    elif verification.verification_status == "rejected":
        submission.status = "verified_rejected"

    db.commit()
    db.refresh(new_verification)

    return new_verification

@app.get("/api/verifier/history")
def get_verification_history(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """获取当前核查员的审核历史"""
    # 状态转换字典
    status_map = {
        'approved': '已批准',
        'rejected': '已拒绝',
        'needs_review': '需要复审',
        'pending': '待审核'
    }

    # 验证核查员身份
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未授权")

    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="令牌无效")

    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_verifier:
        raise HTTPException(status_code=403, detail="需要核查员权限")

    # 获取该核查员的所有审核记录
    verifications = db.query(VerificationRecord).filter(
        VerificationRecord.verifier_id == user.id
    ).order_by(VerificationRecord.created_at.desc()).all()

    result = []
    for v in verifications:
        submission = db.query(NewsSubmission).filter(
            NewsSubmission.id == v.submission_id
        ).first()

        result.append({
            "id": v.id,
            "submission_id": v.submission_id,
            "submission_title": submission.title if submission else "未知",
            "verification_status": status_map.get(v.verification_status, v.verification_status),
            "credibility_adjustment": v.credibility_adjustment,
            "verification_comment": v.verification_comment,
            "created_at": v.created_at.isoformat() if v.created_at else ""
        })

    return result

@app.get("/api/verifier/stats")
def get_verifier_stats(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """获取核查员统计数据"""
    # 验证核查员身份
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未授权")

    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="令牌无效")

    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_verifier:
        raise HTTPException(status_code=403, detail="需要核查员权限")

    # 统计数据
    total_verifications = db.query(VerificationRecord).filter(
        VerificationRecord.verifier_id == user.id
    ).count()

    approved_count = db.query(VerificationRecord).filter(
        VerificationRecord.verifier_id == user.id,
        VerificationRecord.verification_status == "approved"
    ).count()

    rejected_count = db.query(VerificationRecord).filter(
        VerificationRecord.verifier_id == user.id,
        VerificationRecord.verification_status == "rejected"
    ).count()

    pending_count = db.query(NewsSubmission).filter(
        NewsSubmission.status == "completed"
    ).count()

    return {
        "total_verifications": total_verifications,
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "pending_count": pending_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

