from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.constants import Constants
 
load_dotenv()

llm = ChatGroq(model=Constants.llm_model, temperature=Constants.llm_temperature)