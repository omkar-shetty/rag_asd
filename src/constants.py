import os

class Constants:
    llm_model = "openai/gpt-oss-20b"
    llm_temperature = 0
    USE_MCP = os.getenv("USE_MCP") == "1"
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")

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

