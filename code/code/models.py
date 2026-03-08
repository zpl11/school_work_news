from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# ==================== 用户角色表（拆分为三个独立表） ====================

class NormalUser(Base):
    """普通用户表 - 提交新闻进行核查的用户"""
    __tablename__ = "normal_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    phone = Column(String, nullable=True)  # 手机号
    avatar = Column(String, nullable=True)  # 头像路径
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)  # 最后登录时间


class Verifier(Base):
    """核查员表 - 负责审核新闻的专业人员"""
    __tablename__ = "verifiers"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    real_name = Column(String, nullable=True)  # 真实姓名
    department = Column(String, nullable=True)  # 所属部门
    expertise = Column(String, nullable=True)  # 专业领域
    certification = Column(String, nullable=True)  # 资质证书
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


class Admin(Base):
    """管理员表 - 系统管理人员"""
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    real_name = Column(String, nullable=True)  # 真实姓名
    permission_level = Column(Integer, default=1)  # 权限等级：1普通管理员，2超级管理员
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


# 兼容旧代码的User类（作为视图或辅助类）
class User(Base):
    """用户总表 - 用于兼容旧代码和统一登录"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")  # user, verifier, admin
    role_id = Column(Integer, nullable=True)  # 对应角色表中的ID
    is_admin = Column(Boolean, default=False)
    is_verifier = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    submissions = relationship("NewsSubmission", back_populates="user")
    verifications = relationship("VerificationRecord", back_populates="verifier")

class NewsSubmission(Base):
    __tablename__ = "news_submissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, index=True)
    content = Column(Text)
    file_path = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, analyzing, completed
    is_self_check = Column(Boolean, default=False)  # 新增：是否为自检模式
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="submissions")
    analysis = relationship("AnalysisResult", back_populates="submission", uselist=False)

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("news_submissions.id"))
    text_score = Column(Float, default=0.0)
    image_score = Column(Float, default=0.0)
    video_score = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)
    analysis_details = Column(Text)
    text_details = Column(Text)  # 文本分析详细指标
    image_details = Column(Text)  # 图像分析详细指标
    video_details = Column(Text)  # 视频分析详细指标
    created_at = Column(DateTime, default=datetime.utcnow)

    submission = relationship("NewsSubmission", back_populates="analysis")

class AnalysisModel(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    model_type = Column(String)  # text, image, video, audio
    version = Column(String)
    accuracy = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("news_submissions.id"))
    task_type = Column(String)  # review, annotation, verification
    status = Column(String, default="pending")  # pending, in_progress, completed, approved, rejected
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    history = relationship("TaskHistory", back_populates="task")

class TaskHistory(Base):
    __tablename__ = "task_history"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    action = Column(String)  # created, assigned, approved, rejected, commented
    operator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("Task", back_populates="history")

class VerificationRecord(Base):
    """信息核查员的审核记录"""
    __tablename__ = "verification_records"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("news_submissions.id"))
    verifier_id = Column(Integer, ForeignKey("users.id"))

    # 审核结果
    verification_status = Column(String)  # approved, rejected, needs_review
    credibility_adjustment = Column(Float, nullable=True)  # 核查员调整后的可信度评分

    # 审核意见
    verification_comment = Column(Text)  # 核查员的专业意见
    evidence_links = Column(Text, nullable=True)  # 证据链接（JSON格式）

    # 新增：问题标注字段
    has_issue = Column(Boolean, default=False)  # 是否存在问题（错检或漏检）
    issue_type = Column(String, nullable=True)  # 问题类型：false_positive(错检), false_negative(漏检), normal(正常)
    issue_description = Column(Text, nullable=True)  # 问题详细描述
    correct_label = Column(String, nullable=True)  # 正确的标签（真实/虚假）
    correct_score = Column(Float, nullable=True)  # 正确的评分

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    verifier = relationship("User", back_populates="verifications")
    submission = relationship("NewsSubmission")

