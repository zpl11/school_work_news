"""
重置管理员密码
"""
from database import SessionLocal
from models import User
import bcrypt

db = SessionLocal()

# 查找管理员
admin = db.query(User).filter(User.username == 'admin').first()

if admin:
    # 重置密码为 admin123
    new_password = "admin123"
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    admin.hashed_password = hashed
    db.commit()
    print(f"✅ 管理员密码已重置为: {new_password}")
    print(f"   用户名: {admin.username}")
    print(f"   邮箱: {admin.email}")
else:
    print("❌ 管理员账号不存在")

db.close()

