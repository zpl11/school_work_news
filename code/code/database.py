from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config import settings
from models import Base

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """初始化数据库"""
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库初始化完成")

    # 初始化默认数据
    _init_default_data()

def _init_default_data():
    """初始化默认数据"""
    from models import User, AnalysisModel, Admin, Verifier, NormalUser
    from auth import hash_password

    db = SessionLocal()

    # ==================== 创建管理员 ====================
    admin_record = db.query(Admin).filter(Admin.username == "admin").first()
    if not admin_record:
        admin_record = Admin(
            username="admin",
            email="admin@example.com",
            hashed_password=hash_password("123456"),
            real_name="系统管理员",
            permission_level=2
        )
        db.add(admin_record)
        db.flush()

    admin = db.query(User).filter(User.username == "admin").first()
    if not admin:
        admin = User(
            username="admin",
            email="admin@example.com",
            hashed_password=hash_password("123456"),
            role="admin",
            role_id=admin_record.id if admin_record else None,
            is_admin=True,
            is_verifier=False
        )
        db.add(admin)
        db.flush()

    # ==================== 创建核查员 ====================
    verifier_record = db.query(Verifier).filter(Verifier.username == "verifier").first()
    if not verifier_record:
        verifier_record = Verifier(
            username="verifier",
            email="verifier@example.com",
            hashed_password=hash_password("verifier123"),
            real_name="张核查",
            department="新闻核查部",
            expertise="社会新闻"
        )
        db.add(verifier_record)
        db.flush()

    verifier = db.query(User).filter(User.username == "verifier").first()
    if not verifier:
        verifier = User(
            username="verifier",
            email="verifier@example.com",
            hashed_password=hash_password("verifier123"),
            role="verifier",
            role_id=verifier_record.id if verifier_record else None,
            is_admin=False,
            is_verifier=True
        )
        db.add(verifier)
        db.flush()

    # ==================== 创建普通用户 ====================
    normal_user_record = db.query(NormalUser).filter(NormalUser.username == "testuser").first()
    if not normal_user_record:
        normal_user_record = NormalUser(
            username="testuser",
            email="testuser@example.com",
            hashed_password=hash_password("test123"),
            phone="13800138000"
        )
        db.add(normal_user_record)
        db.flush()

    testuser = db.query(User).filter(User.username == "testuser").first()
    if not testuser:
        testuser = User(
            username="testuser",
            email="testuser@example.com",
            hashed_password=hash_password("test123"),
            role="user",
            role_id=normal_user_record.id if normal_user_record else None,
            is_admin=False,
            is_verifier=False
        )
        db.add(testuser)
        db.flush()

    # 创建默认模型
    models = [
        {"name": "BERT-Text", "model_type": "text", "version": "1.0", "accuracy": 0.92},
        {"name": "ResNet-Image", "model_type": "image", "version": "1.0", "accuracy": 0.88},
        {"name": "3D-CNN-Video", "model_type": "video", "version": "1.0", "accuracy": 0.85},
        {"name": "Wav2Vec-Audio", "model_type": "audio", "version": "1.0", "accuracy": 0.83},
    ]

    for model_data in models:
        existing = db.query(AnalysisModel).filter(AnalysisModel.name == model_data["name"]).first()
        if not existing:
            model = AnalysisModel(**model_data)
            db.add(model)

    db.commit()
    db.close()

