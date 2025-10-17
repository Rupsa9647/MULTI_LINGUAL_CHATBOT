# generator.py
import os
import sqlite3
from retriever import Retriever
from reranker import Reranker
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# Load environment variables from .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")

# Set it for safety (LangChain uses it automatically)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

DB_PATH = "chat_history.db"
#GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)


class Generator:
    def __init__(self, use_reranker: bool = True):
        # ----------------------------
        # GROQ API key validation
        if not os.getenv("GOOGLE_API_KEY"):
         raise ValueError("❌ GOOGLE_API_KEY not set in environment variables.")

# ----------------------------
# LLM & Memory setup (Gemini)
# ----------------------------
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # Free and fast model
        temperature=0.3,
        max_output_tokens=512,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="query",
            return_messages=True
        )

        # ----------------------------
        # DB setup & load previous history
        # ----------------------------
        self._init_db()
        self._load_history()

        # ----------------------------
        # Reranker
        # ----------------------------
        self.use_reranker = use_reranker
        if self.use_reranker:
            self.reranker = Reranker()

        # ----------------------------
        # LangChain Prompt & Chain
        # ----------------------------
        self.chain = self._init_chain()

    # ----------------------------
    # Database functions
    # ----------------------------
    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT,
                bot_response TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_history(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_message, bot_response FROM chat_history")
        rows = cursor.fetchall()
        conn.close()
        for user_msg, bot_msg in rows:
            self.memory.chat_memory.add_user_message(user_msg)
            self.memory.chat_memory.add_ai_message(bot_msg)

    def _save_chat(self, user_message: str, bot_response: str):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_message, bot_response) VALUES (?, ?)",
            (user_message, bot_response)
        )
        conn.commit()
        conn.close()

    # ----------------------------
    # Prompt & Chain
    # ----------------------------
    def _init_chain(self):
        prompt_template = ChatPromptTemplate.from_template("""
        You are a multilingual expert assistant. Use the following context and prior chat history 
        to answer the user's question accurately and concisely.

        Context:
        {context}

        Chat History:
        {chat_history}

        User Question: {query}

        Answer:
        """)
        return LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            memory=self.memory
        )

    # ----------------------------
    # Generate answer
    # ----------------------------
    def generate_answer(self, query: str, retriever_results: list):
        """
        Generate answer from query and retrieved results.
        retriever_results: list of dicts from Retriever.hybrid_search
        """
        if not retriever_results:
            return "⚠ No relevant context found for this query."

        # ----------------------------
        # Optional Reranker
        # ----------------------------
        context_results = retriever_results
        if self.use_reranker and hasattr(self.reranker, "rerank_results"):
            context_results = self.reranker.rerank_results(query, retriever_results)

        # Join top N chunks into context
        context = "\n\n".join([r["text"] for r in context_results[:5]])

        if not context.strip():
            return "⚠ No relevant context found for this query."

        # Invoke LangChain
        result = self.chain.invoke({"context": context, "query": query})

        # Safe extraction of response text
        if isinstance(result, dict):
            bot_response = result.get("text") or result.get("output_text", "")
        else:
            bot_response = str(result)

        # Save to DB
        self._save_chat(query, bot_response)
        return bot_response
