import pandas as pd
from azure.cosmos import CosmosClient
import uuid

# Replace with your values
COSMOS_DB_URI = ""
COSMOS_DB_KEY = ""
DATABASE_NAME = "incident-db"
CONTAINER_NAME = "incidents"
EXCEL_FILE_PATH = ""  # <-- Update this

try:
    # Assuming headers are in the 2nd row (index 1)
    df = pd.read_excel(EXCEL_FILE_PATH, header=1)
    print(f"✅ Successfully read Excel file from: {EXCEL_FILE_PATH}")
except FileNotFoundError:
    print(
        f"❌ Error: Excel file not found at {EXCEL_FILE_PATH}. Please check the path.")
    exit()

# Fill NaNs with empty string (optional)
df = df.fillna("")

# Connect to Cosmos DB
try:
    client = CosmosClient(COSMOS_DB_URI, credential=COSMOS_DB_KEY)
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)
    print(
        f"✅ Connected to Cosmos DB database '{DATABASE_NAME}', container '{CONTAINER_NAME}'")
except Exception as e:
    print(f"❌ Error connecting to Cosmos DB: {e}")
    exit()

# Insert each row as a document
print("🚀 Starting data upload to Cosmos DB...")
uploaded_count = 0
for _, row in df.iterrows():
    item = row.to_dict()
    item['id'] = str(uuid.uuid4())  # Cosmos DB requires unique 'id'

    try:
        container.upsert_item(item)
        uploaded_count += 1
    except Exception as e:
        print(f"❌ Error uploading item: {e}")

print(f"---")
print(
    f"✅ Excel data successfully uploaded to Cosmos DB! Total items uploaded: {uploaded_count}")
print("---")
