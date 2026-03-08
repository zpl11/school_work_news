"""
数据库迁移脚本 - 添加自检和问题标注字段
"""
from database import SessionLocal, engine
from sqlalchemy import text

def migrate_database():
    """迁移数据库，添加新字段"""
    db = SessionLocal()
    
    try:
        print("🔄 开始数据库迁移...")
        
        # 1. 为 news_submissions 表添加 is_self_check 字段
        try:
            db.execute(text("""
                ALTER TABLE news_submissions 
                ADD COLUMN is_self_check BOOLEAN DEFAULT 0
            """))
            print("✅ 添加 news_submissions.is_self_check 字段")
        except Exception as e:
            if "duplicate column name" in str(e).lower():
                print("⚠️ news_submissions.is_self_check 字段已存在")
            else:
                print(f"❌ 添加 news_submissions.is_self_check 失败: {e}")
        
        # 2. 为 verification_records 表添加问题标注字段
        fields_to_add = [
            ("has_issue", "BOOLEAN DEFAULT 0"),
            ("issue_type", "VARCHAR"),
            ("issue_description", "TEXT"),
            ("correct_label", "VARCHAR"),
            ("correct_score", "FLOAT")
        ]
        
        for field_name, field_type in fields_to_add:
            try:
                db.execute(text(f"""
                    ALTER TABLE verification_records 
                    ADD COLUMN {field_name} {field_type}
                """))
                print(f"✅ 添加 verification_records.{field_name} 字段")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    print(f"⚠️ verification_records.{field_name} 字段已存在")
                else:
                    print(f"❌ 添加 verification_records.{field_name} 失败: {e}")
        
        db.commit()
        print("✅ 数据库迁移完成！")
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate_database()

