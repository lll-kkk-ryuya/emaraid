import os
import chromadb
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core import get_response_synthesizer
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from src.router.room.chat.nodes import DocumentProcessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import PromptTemplate

load_dotenv()
# 環境変数 'OPENAI_API_KEY' を取得
openai_api_key = os.getenv('OPENAI_API_KEY')

class VectorStoreAndQueryEngine:
    def __init__(self, path="emaraid_db",document_directory=None):
        self.document_directory = document_directory
        self.vector_query_engines = {}
        self.path=path

    def initialize_vector_store_index(self, collection_name, nodes=None, embed_batch_size=64):
        embed_model = OpenAIEmbedding(embed_batch_size=embed_batch_size ,api_key=openai_api_key )
        db = chromadb.PersistentClient(path=self.path)
        for node in nodes:
            if 'entities' in node.metadata and isinstance(node.metadata['entities'], list):
                entities_str = ', '.join(node.metadata['entities'])
                node.metadata['entities'] = entities_str
        try:
            chroma_collection = db.get_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
            index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context, storage_context=storage_context)
            return collection_name, index
        except ValueError as e:
            print(e)
            chroma_collection = db.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
            index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, service_context=service_context)


    def initialize_vector_query_engine(self, index, model="gpt-4", temperature=0.1, similarity_top_k=5):
        llm = OpenAI(model=model, temperature=temperature,api_key=openai_api_key )
        service_context = ServiceContext.from_defaults(llm=llm)
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
        text_qa_template_str = (
    "以下にコンテキスト情報があります。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "コンテキスト情報に基づいて、事前知識に頼らずに回答してください。\n"
    "回答は必ず日本語でお願いします。\n"
    "タメ口で喋って!「だよ!」とか「てね！」とかの語尾をいい感じに使って。カジュアルな専門インストラクターとして、超フランクで話しやすい感じでよろしく。\n"
    "質問: {query_str}\n"
    "回答: "
)
        text_qa_template = PromptTemplate(text_qa_template_str)
        response_synthesizer = get_response_synthesizer(
        #response_mode="refine",
        service_context=service_context, 
        streaming=True,
        text_qa_template=text_qa_template,
        use_async=True
    )
        #rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
        vector_query_engine = RetrieverQueryEngine.from_args(
            retriever=vector_retriever, 
            response_synthesizer=response_synthesizer, 
            #node_postprocessors=[rerank]
            use_async = True,
            )
        return vector_query_engine

    def add_vector_query_engine(self, collection_name, model="gpt-3.5-turbo", temperature=0.4, similarity_top_k=5):      
        document_processor = DocumentProcessor(directory=self.document_directory, model=model)  
        nodes = document_processor.process_documents()
        _,index = self.initialize_vector_store_index(collection_name,nodes)
        vector_query_engine = self.initialize_vector_query_engine(index, model, temperature, similarity_top_k)
        #self.vector_query_engines[collection_name] = vector_query_engine
        return vector_query_engine


#directory = "path/to/your/documents"
# collection_name = "your_collection_name"

# VectorQueryEngineManager インスタンスの作成
# vector_query_engine_manager = VectorQueryEngineManager(directory)

# 特定のコレクションに対するベクタークエリエンジンの初期化と追加
# vector_query_engine_manager.add_vector_query_engine(collection_name, model="gpt-4", temperature=0.4, similarity_top_k=5)

# 必要に応じて、追加のコレクションや異なる設定でベクタークエリエンジンを追加できます