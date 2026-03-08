from database import SessionLocal
from models import AnalysisResult

db = SessionLocal()

# 获取表的所有列
columns = [c.name for c in AnalysisResult.__table__.columns]
print("AnalysisResult 表的字段:")
for col in columns:
    print(f"  - {col}")

# 查看一个实际的记录
result = db.query(AnalysisResult).first()
if result:
    print("\n示例记录:")
    for col in columns:
        value = getattr(result, col, None)
        print(f"  {col}: {value}")
else:
    print("\n没有分析结果记录")

db.close()

