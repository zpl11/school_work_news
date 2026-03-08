"""
添加详细指标字段到 analysis_results 表
"""
from database import engine
from sqlalchemy import text

# 添加新字段
with engine.connect() as conn:
    try:
        # 添加 text_details 字段
        conn.execute(text("ALTER TABLE analysis_results ADD COLUMN text_details TEXT"))
        print("✅ 添加 text_details 字段成功")
    except Exception as e:
        print(f"⚠️ text_details 字段可能已存在: {e}")
    
    try:
        # 添加 image_details 字段
        conn.execute(text("ALTER TABLE analysis_results ADD COLUMN image_details TEXT"))
        print("✅ 添加 image_details 字段成功")
    except Exception as e:
        print(f"⚠️ image_details 字段可能已存在: {e}")
    
    try:
        # 添加 video_details 字段
        conn.execute(text("ALTER TABLE analysis_results ADD COLUMN video_details TEXT"))
        print("✅ 添加 video_details 字段成功")
    except Exception as e:
        print(f"⚠️ video_details 字段可能已存在: {e}")
    
    try:
        # 添加 audio_details 字段
        conn.execute(text("ALTER TABLE analysis_results ADD COLUMN audio_details TEXT"))
        print("✅ 添加 audio_details 字段成功")
    except Exception as e:
        print(f"⚠️ audio_details 字段可能已存在: {e}")
    
    conn.commit()
    print("\n✅ 数据库迁移完成！")

