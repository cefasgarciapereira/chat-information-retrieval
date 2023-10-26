from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from translate import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
import os

MODEL_PATH = "./models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

huggingfacehub_api_token="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"

nltk.download('punkt')
nltk.download('stopwords')

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.1, "max_new_tokens": 150})

documents = []

folder = "./docs"

for filename in os.listdir(folder):
  f_content = open(folder+"/"+filename, "r").read()
  documents.append(f_content)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens if w.isalnum()] 
    tokens = [w for w in tokens if w not in nltk.corpus.stopwords.words('english')]
    return ' '.join(tokens)

vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
tfidf_matrix = vectorizer.fit_transform(documents)

def get_context(question):
    question_tfidf = vectorizer.transform([question])
    similarities = cosine_similarity(question_tfidf, tfidf_matrix)
    document_index = similarities.argmax()
    return documents[document_index]

def get_answer(context, question):
  template =  """
    Use the following context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    {context}
    Question: {question}
    Helpful Answer:
  """
  prompt = PromptTemplate.from_template(template)
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response = llm_chain.run({"context": context, "question": question})
  return response

def make_a_question(question):
  context = get_context(question)
  answer = get_answer(context, question)
  return (answer, context)