from flask import Flask, render_template, request, jsonify
from googlesearch import search

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]= "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"
import warnings
warnings.filterwarnings("ignore")
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
import textwrap 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

app = Flask(__name__)

loader = TextLoader("data.txt")
document = loader.load()

#pre-processing
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('/n')
    wrapped_lines = [textwrap.fill(line, width=width)for line in lines]
    wraped_text = '/n'.join(wrapped_lines)
    return wraped_text


#text splitting 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap = 0)
docs = text_splitter.split_documents(document)



#embedding
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)


#Q-A
llm=HuggingFaceHub(repo_id='google/flan-t5-xxl', model_kwargs={"temperature" : 0.8, "max_length" : 512})
chain = load_qa_chain(llm, chain_type="stuff")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    if request.method == "POST":
        query_text = request.form["query"]

        if not query_text:
            return jsonify({"input": query_text, "output": "Sorry, I didn't understand what you said. How may I assist you today?"})

        # Perform similarity search
        docs_results = db.similarity_search(query_text)

        # Run the chain
        output = chain.run(input_documents=docs_results, question=query_text)

        if not output:
            # If the chatbot doesn't have an answer, try searching on Google
            google_results = list(search(query_text, num=1, stop=1))
            if google_results:
                output = f"Google says: {google_results[0]}"
            else:
                output = "Sorry, I'm unable to provide information related to the question you asked. Please try something else."

        return jsonify({"input": query_text, "output": output})

if __name__ == "__main__":
    app.run(debug=True)
