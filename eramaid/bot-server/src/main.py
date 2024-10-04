import os
from supabase import create_client, Client
from cors_config import add_cors_middleware
from llama_index.core.base.response.schema import StreamingResponse,AsyncStreamingResponse,Response
from starlette.responses import Response as StarletteResponse
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# 正しいrouterインスタンスのインポート
from src.router.room.chat.mainbot import QueryService
import json
from typing import Optional
from uuid import uuid4, UUID
import asyncio
from starlette.websockets import WebSocketDisconnect ,WebSocketState
import os
import time
import logging
import traceback
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.schema import QueryBundle
from src.router.room.chat.vector_engines import VectorStoreAndQueryEngine
import nest_asyncio 
nest_asyncio.apply()
openai_api_key = os.getenv('OPENAI_API_KEY')
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

app = FastAPI()
add_cors_middleware(app)


class QueryRequest(BaseModel):
    user_id: Optional[str] = None  # ユーザーID
    chatroom_id: Optional[str] = None # チャットルームID
    message: str  # ユーザーからのメッセージ

class ResponseModel(BaseModel):
    prompt_id: str
    reply_from_bot: str
    createdAt: str

query_engine = None
app_ready = False
query_gen_prompt = None
@app.on_event("startup")
async def startup_event():
    global query_engine, app_ready,query_gen_prompt,query_service
    db_url = 'sqlite:///example.db'
    collection_names = [
    "text.pdf",
]
    table_name = "all_curce"
    tool_metadata = {
    "text.pdf": {
        "name": "栄養士データ",
        "description": "基本的な内容はここから参照"
    },

}
    
    query_gen_str = """\
You are a helpful assistant that generates multiple search queries based on a \
single input query. Generate {num_queries} search queries, one on each line, 
生徒の目線に立って生成を行うこと。\
related to the following input query.\
また、文末に"を教えて。"で終えれるよな文に変換するようにしなさい。\
Query: {query}
Queries:
**If the text is inappropriate, convert it to an appropriate text.**
"""
    #query_gen_prompt = PromptTemplate(query_gen_str)
    #query_service = QueryService(db_url,collection_names,table_name) 
    #await query_service.setup_engines()
    #query_engine = await query_service.query_engine()
    query_gen_prompt = PromptTemplate(query_gen_str)
    vector_store_query_engine = VectorStoreAndQueryEngine()
    query_engine = vector_store_query_engine.add_vector_query_engine(collection_name="text.pdf", model="gpt-4", temperature=0.4, similarity_top_k=5)
    print("起動")
    app_ready = True  # アプリケーションの準備が完了

@app.get("/health")
def health_check():
    if app_ready:
        return {"status": "ok"}
    else:
        return StarletteResponse(content={"status": "starting"}, status_code=503)

def generate_queries(query: str, llm, num_queries: int = 2):
    response = llm.predict(
        query_gen_prompt, num_queries=num_queries, query=query
    )
    # assume LLM proper put each query on a newline
    queries = response.split("\n")
    queries_str = "\n".join(queries)
    print(f"Generated queries:\n{queries_str}")
    return queries

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global query_engine, query_gen_prompt,query_service
    await websocket.accept()
    logging.debug("WebSocket connection accepted.")

    try:
        while True:
            start_time = time.time()  # 実行開始時刻を記録
            data = await websocket.receive_text()
            logging.debug(f"Received data: {data}")

            data = json.loads(data)
            message_id = data['id']
            query_str = data.get('content')
            logging.debug(f"Processing query: {query_str}")
            llm_query = OpenAI(model="gpt-4", api_key=openai_api_key)
            queries = generate_queries(query_str, llm_query)
            first_query = QueryBundle(query_str=queries[1])
            result = await query_engine._aquery(first_query)
            #result =  query_engine._query(first_query)
            #result = await query_engine.achat(first_query)
            #result = query_engine.chat(first_query)
            print(type(result))
            if isinstance(result, StreamingResponse):
                async for text in result.response_gen:
                    reply_json_str = json.dumps({"id": message_id, "reply_from_bot": text}, ensure_ascii=False)
                    logging.debug(f"Sending streaming response: {reply_json_str}")
                    await websocket.send_text(reply_json_str)
                end_of_stream_message = json.dumps({"id": message_id, "reply_from_bot": "STREAM_END", "status": "completed"})
                await websocket.send_text(end_of_stream_message)
                logging.debug("All parts of the streaming response have been sent.")

            if isinstance(result, AsyncStreamingResponse):
                async for text in result.async_response_gen:#デプロイする際はなぜか()が必要
                    reply_json_str = json.dumps({"id": message_id, "reply_from_bot": text}, ensure_ascii=False)
                    logging.debug(f"Sending streaming response: {reply_json_str}")
                    await websocket.send_text(reply_json_str)
                end_of_stream_message = json.dumps({"id": message_id, "reply_from_bot": "STREAM_END", "status": "completed"})
                await websocket.send_text(end_of_stream_message)
                logging.debug("All parts of the streaming response have been sent.")
            end_time = time.time()  # 実行終了時刻を記録
            total_time = end_time - start_time  # 実行時間を計算
            print(f"Total execution time: {total_time} seconds")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        if websocket.client_state == WebSocketState.DISCONNECTED:
            await websocket.close()
            logging.debug("WebSocket connection closed due to an error.")


     
@app.websocket("/ws_test")
async def websocket_endpoint(websocket: WebSocket):
    global query_engine, query_gen_prompt,query_service
    await websocket.accept()
    logging.debug("WebSocket connection accepted.")

    try:
        while True:
            start_time = time.time()  # 実行開始時刻を記録
            data = await websocket.receive_text()
            logging.debug(f"Received data: {data}")

            data = json.loads(data)
            message_id = data['id']
            query_str = data.get('content')
            logging.debug(f"Processing query: {query_str}")
            llm_query = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
            queries = generate_queries(query_str, llm_query)
            first_query = QueryBundle(query_str=queries[1])
            result = await query_engine._aquery(first_query)
            #result =  query_engine._query(first_query)
            #result = await query_engine.achat(first_query)
            #result = query_engine.chat(first_query)
            print(type(result))
            if isinstance(result, StreamingResponse):
                for text in result.response_gen:
                    reply_json_str = json.dumps({"id": message_id, "reply_from_bot": text}, ensure_ascii=False)
                    logging.debug(f"Sending streaming response: {reply_json_str}")
                    await websocket.send_text(reply_json_str)
                end_of_stream_message = json.dumps({"id": message_id, "reply_from_bot": "STREAM_END", "status": "completed"})
                await websocket.send_text(end_of_stream_message)
                logging.debug("All parts of the streaming response have been sent.")

            if isinstance(result, AsyncStreamingResponse):
                async for text in result.async_response_gen():#デプロイする際はなぜか()が必要
                    reply_json_str = json.dumps({"id": message_id, "reply_from_bot": text}, ensure_ascii=False)
                    logging.debug(f"Sending streaming response: {reply_json_str}")
                    await websocket.send_text(reply_json_str)
                end_of_stream_message = json.dumps({"id": message_id, "reply_from_bot": "STREAM_END", "status": "completed"})
                await websocket.send_text(end_of_stream_message)
                logging.debug("All parts of the streaming response have been sent.")
            end_time = time.time()  # 実行終了時刻を記録
            total_time = end_time - start_time  # 実行時間を計算
            print(f"Total execution time: {total_time} seconds")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        if websocket.client_state == WebSocketState.DISCONNECTED:
            await websocket.close()
            logging.debug("WebSocket connection closed due to an error.")

