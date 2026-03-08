from database import init_db, SessionLocal
from models import User, NormalUser, Verifier, Admin, AnalysisModel, Task, TaskHistory
from auth import hash_password

def init_default_data():
    """初始化默认数据"""
    db = SessionLocal()

    # 创建表
    init_db()

    # ==================== 创建管理员 ====================
    # 1. 在Admin表中创建
    admin_record = db.query(Admin).filter(Admin.username == "admin").first()
    if not admin_record:
        admin_record = Admin(
            username="admin",
            email="admin@example.com",
            hashed_password=hash_password("123456"),
            real_name="系统管理员",
            permission_level=2  # 超级管理员
        )
        db.add(admin_record)
        db.flush()
        print("✅ 创建管理员账号(Admin表): admin / 123456")

    # 2. 在User总表中也创建（兼容旧代码）
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
        print("✅ 创建管理员账号(User表): admin / 123456")

    # ==================== 创建核查员 ====================
    # 1. 在Verifier表中创建
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
        print("✅ 创建核查员账号(Verifier表): verifier / verifier123")

    # 2. 在User总表中也创建
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
        print("✅ 创建核查员账号(User表): verifier / verifier123")

    # ==================== 创建普通用户 ====================
    # 1. 在NormalUser表中创建
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
        print("✅ 创建测试用户(NormalUser表): testuser / test123")

    # 2. 在User总表中也创建
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
        print("✅ 创建测试用户(User表): testuser / test123")

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
    print("✅ 数据库初始化完成，默认数据已创建")
    print("\n📊 数据表清单:")
    print("  1. normal_users - 普通用户表")
    print("  2. verifiers - 核查员表")
    print("  3. admins - 管理员表")
    print("  4. users - 用户总表(兼容)")
    print("  5. news_submissions - 新闻提交表")
    print("  6. analysis_results - 分析结果表")
    print("  7. models - 模型表")
    print("  8. tasks - 任务表")
    print("  9. task_history - 任务历史表")
    print("  10. verification_records - 审核记录表")

if __name__ == "__main__":
    init_default_data()

