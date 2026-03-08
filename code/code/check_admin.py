"""
检查并重置管理员账号
"""
from database import SessionLocal
from models import User
from auth import hash_password, verify_password

db = SessionLocal()

print("=" * 60)
print("  检查管理员账号")
print("=" * 60)

# 查找管理员
admin = db.query(User).filter(User.username == 'admin').first()

if admin:
    print(f"\n✅ 找到管理员账号:")
    print(f"   用户名: {admin.username}")
    print(f"   邮箱: {admin.email}")
    print(f"   是否管理员: {admin.is_admin}")
    print(f"   是否核查员: {admin.is_verifier}")
    print(f"   创建时间: {admin.created_at}")
    
    # 测试密码
    print(f"\n🔐 测试密码:")
    test_passwords = ["123456", "admin123", "admin", "password"]
    
    for pwd in test_passwords:
        if verify_password(pwd, admin.hashed_password):
            print(f"   ✅ 当前密码是: {pwd}")
            break
    else:
        print(f"   ❌ 当前密码不是常用密码")
        print(f"\n🔧 重置密码为: 123456")
        admin.hashed_password = hash_password("123456")
        db.commit()
        print(f"   ✅ 密码重置成功！")
else:
    print("\n❌ 管理员账号不存在，正在创建...")
    admin = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hash_password("123456"),
        is_admin=True,
        is_verifier=False
    )
    db.add(admin)
    db.commit()
    print(f"   ✅ 管理员账号创建成功！")

print("\n" + "=" * 60)
print("  管理员登录信息")
print("=" * 60)
print(f"  用户名: admin")
print(f"  密码: 123456")
print(f"  访问地址: http://localhost:8000/admin")
print("=" * 60)

db.close()

