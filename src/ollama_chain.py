from langchain_ollama import ChatOllama as Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory
from src.utils import load_config
from src.vectorstore import VectorDB

class OllamaChain:
    def __init__(self, chat_memory) -> None:
        config = load_config()
        prompt = PromptTemplate(
            template="""History: {chat_history}\nQuestion: {input}\nAnswer:""",
            input_variables=['chat_history', 'input']
        )
        self.memory = ConversationBufferWindowMemory(
            memory_key='chat_history', chat_memory=chat_memory, k=3, return_messages=True
        )
        llm = Ollama(**config['chat_model'])
        # Simple LLM Chain
        from langchain_classic.chains import LLMChain
        self.llm_chain = LLMChain(prompt=prompt, llm=llm, memory=self.memory, output_parser=StrOutputParser())

    def run(self, user_input):
        return self.llm_chain.invoke({"input": user_input})['text']

class OllamaRAGChain:
    def __init__(self, chat_memory, uploaded_file=None):
        config = load_config()
        self.chat_memory = chat_memory
        self.llm = Ollama(**config['chat_model'])
        self.vector_db = VectorDB('chroma', 'any')
        self.retriever = None

        if uploaded_file:
            # 1. Indexing the PDF
            self.vector_db.index(uploaded_file)
            self.retriever = self.vector_db.as_retriever()

    def run(self, user_input):
        if not self.retriever:
            return "RAG System ready nahi hai. File upload karein."

        # 1. Manual Context Retrieval
        # Hum khud documents nikaal rahe hain taake LangChain crash na ho
        docs = self.retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Build Manual Prompt
        # Chat history ko text mein convert karna
        history_text = ""
        for msg in self.chat_memory.messages[-5:]: # Last 5 messages
            role = "User" if msg.type == "human" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        prompt = f"""You are a helpful assistant. Use the context below to answer the user's question. 
        If you don't know, just say you don't know.

        Context:
        {context}

        History:
        {history_text}

        User Question: {user_input}
        Assistant:"""

        # 3. Direct LLM Call
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        # 4. Update Memory Manually
        self.chat_memory.add_user_message(user_input)
        self.chat_memory.add_ai_message(answer)

        return answer