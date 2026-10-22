import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import engine
from db.models import Base

Base.metadata.create_all(engine)
print("Таблицы созданы")