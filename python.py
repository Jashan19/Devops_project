# fetch_and_validate_data.py

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# 1️⃣ Initialize Firebase
cred = credentials.Certificate("ServiceAccount.json")
firebase_admin.initialize_app(cred)

# 2️⃣ Connect to Firestore
db = firestore.client()

# 3️⃣ Fetch data from 'journals' collection
docs = db.collection("journals").stream()
data = [doc.to_dict() for doc in docs]

valid_data = []
invalid_data = []

# 4️⃣ Validate journal NLP schema
for d in data:
    if not d or not d.get("text"):
        continue

    valid_data.append({
        "text": d["text"],
        "date": str(d.get("date", "")),
        "updatedAt": str(d.get("updatedAt", ""))
    })


    # except AssertionError:
    #     print(f"❌ Invalid record #{i}: {d}")
    #     invalid_data.append(d)

print("✅ Valid records:", len(valid_data))
print("❌ Invalid records:", len(invalid_data))

# 5️⃣ Save clean NLP-ready data
df = pd.DataFrame(valid_data)
df.to_csv("clean_journal_data.csv", index=False)

print("📁 Saved clean_journal_data.csv")
