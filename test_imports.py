# Test each import separately
try:
    from langchain_community.vectorstores import Chroma
    print("✅ langchain_community.vectorstores import successful")
except ImportError as e:
    print("❌ Error importing from langchain_community.vectorstores:", e)

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    print("✅ langchain_openai import successful")
except ImportError as e:
    print("❌ Error importing from langchain_openai:", e)

try:
    from langchain.chains import RetrievalQA
    print("✅ langchain.chains import successful")
except ImportError as e:
    print("❌ Error importing from langchain.chains:", e) 