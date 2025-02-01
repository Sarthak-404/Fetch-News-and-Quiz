import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Flask API!"

loader = WebBaseLoader("https://www.financialexpress.com/")
docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)
vectors = FAISS.from_documents(final_documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """
        Give me the top 10 on the basis of the context provided which is fetched from a news website.
        Don't include headlines, analyze the context, and try to keep the news part only.
        Don't explain what the website section is doing, just tell the important news.
        You can skip the starting and ending few data.
        {context}
    """
)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route('/news', methods=['GET'])
def get_financial_news():
    try:
        response = retrieval_chain.invoke({"input": "Give top 10 news about finance"})
        news_summary = response.get('answer', "No news found")
        return jsonify({"news_summary": news_summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/quiz', methods=['GET'])
def get_quiz():
    try:
        # Fetch latest news first
        response = retrieval_chain.invoke({"input": "Give top 10 news about finance"})
        news_summary = response.get('answer', "No news found")

        # Create Quiz Prompt
        quiz_prompt = ChatPromptTemplate.from_template("""
            Create a multiple choice quiz using the given context. 
            The quiz should focus on how the details in the context affect daily life and more.
            Frame questions about the possible effects and consequences of the information.
            Make sure that options are vibrant and difficulty increases after every question.
            Keep only 5 questions and they can have multiple correct answers. 
            {context}
        """)

        quiz_text = quiz_prompt.invoke({'context': news_summary})
        quiz_response = llm.invoke(quiz_text)

        return jsonify({"quiz": quiz_response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
