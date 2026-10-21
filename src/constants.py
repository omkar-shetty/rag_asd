import os

class Constants:
    llm_model = "openai/gpt-oss-20b"
    llm_temperature = 0
    USE_MCP = os.getenv("USE_MCP") == "1"
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
    DYNAMODB_ENDPOINT = os.getenv("DYNAMODB_ENDPOINT") 
    AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
    GAPS_TABLE = os.getenv("GAPS_TABLE", "corpus_gaps")
    MAX_QUESTION_LEN = 1000
    MAX_URLS_PER_GAP = 5
    MAX_URL_LEN = 200

    CURATED_DOMAINS = [
    "cdc.gov",              
    "nih.gov",              
    "autismspeaks.org",     
    "who.int",              
    "nichd.nih.gov",        
    "autism.org",
    "thinkingautismguide.com",
    "autisticadvocacy.org",
    "ed.gov",
    "medlineplus.gov",
    "mayoclinic.org",          
    ]

