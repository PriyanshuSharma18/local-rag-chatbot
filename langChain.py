import collections
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class langchain:
    def __init__(self) -> None:
        # Templates
        self.text_template = """You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. 
To help you, you can use the following context: {context}

You prefer to use bullet points to summarize.

Question: {question}
"""

        self.img_template = """Reformulate the following prompt that is used to generate images. 
Create a new proper prompt which only keeps the meaningful information and removes words that are not useful to the image's meaning.

The prompt is: {question}
Please return only the new prompt and no surrounding text.
"""

        self.resume_template = """You're a helpful AI assistant tasked to summarize PDF files.
You're friendly and you answer extensively with multiple sentences. 
To answer, please use only the following context: {context}

Question: {question}
"""

        self.query_template = """Based on the following prompt, determine if the user is expecting a text answer, a generated image, or a PDF summary. 
If it is a text answer, return: text. 
If it is a generated image, return: img. 
If it is a resume, return: resume.

The prompt is: {question}
If unsure, say: unknown.
"""

        self.llm = self.get_llm()
        self.retriever = None

    # Determine the query type
    def get_query_type(self, question):
        prompt = self.get_prompt("query")
        chain = prompt | self.llm | StrOutputParser()
        query_type = chain.invoke({"question": question})
        return query_type.strip().lower()

    # Choose the prompt based on query type
    def get_prompt(self, query_type):
        query_type = query_type.strip().lower()

        if query_type == "text":
            prompt = ChatPromptTemplate.from_template(self.text_template)
        elif query_type == "img":
            prompt = ChatPromptTemplate.from_template(self.img_template)
        elif query_type == "resume":
            prompt = ChatPromptTemplate.from_template(self.resume_template)
        elif query_type == "query":
            prompt = ChatPromptTemplate.from_template(self.query_template)
        else:
            print(f"⚠️ Warning: Unknown query_type '{query_type}'. Using 'text' template by default.")
            prompt = ChatPromptTemplate.from_template(self.text_template)

        return prompt

    # ✅ Use llama3 model here
    def get_llm(self):
        return ChatOllama(model="llama3", temperature=0.1)

    # Generate chatbot answer
    def get_chatbot_answer(self, question, context="", query_type="text"):
        query_type = query_type.strip().lower()
        prompt = self.get_prompt(query_type)
        chain = prompt | self.llm | StrOutputParser()

        if query_type in ["text", "resume"]:
            answer = chain.invoke({"context": context, "question": question})
        else:
            answer = chain.invoke({"question": question})

        return answer
