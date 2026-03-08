from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# 用户相关
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    is_admin: bool = False
    is_verifier: bool = False

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_admin: bool
    is_verifier: bool = False
    created_at: datetime

    class Config:
        from_attributes = True

# 新闻提交相关
class NewsSubmissionCreate(BaseModel):
    title: str
    content: str
    is_self_check: bool = False  # 新增：是否为自检模式

class NewsSubmissionResponse(BaseModel):
    id: int
    title: str
    content: str
    status: str
    file_path: str | None = None
    is_self_check: bool = False
    created_at: datetime

    class Config:
        from_attributes = True

# 分析结果相关
class AnalysisResultResponse(BaseModel):
    id: int
    text_score: float
    image_score: float
    video_score: float
    audio_score: float
    overall_score: float
    analysis_details: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# 令牌相关
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# 核查员相关
class VerificationCreate(BaseModel):
    submission_id: int
    verification_status: str  # approved, rejected, needs_review
    credibility_adjustment: Optional[float] = None
    verification_comment: str
    evidence_links: Optional[str] = None
    # 新增：问题标注字段
    has_issue: bool = False
    issue_type: Optional[str] = None  # false_positive, false_negative, normal
    issue_description: Optional[str] = None
    correct_label: Optional[str] = None
    correct_score: Optional[float] = None

class VerificationResponse(BaseModel):
    id: int
    submission_id: int
    verifier_id: int
    verification_status: str
    credibility_adjustment: Optional[float]
    verification_comment: str
    evidence_links: Optional[str]
    has_issue: bool = False
    issue_type: Optional[str] = None
    issue_description: Optional[str] = None
    correct_label: Optional[str] = None
    correct_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

