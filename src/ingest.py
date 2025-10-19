from src.config.db import qdrant_db
from src.config.utils import load_FAQS

faqs = load_FAQS()

print(len(faqs))

qdrant_db.create_collection()

points = qdrant_db.create_points(faqs)

qdrant_db.upsert_points(points)
