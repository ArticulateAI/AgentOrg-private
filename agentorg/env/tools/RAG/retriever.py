import logging
import os

from agentorg.env.prompts import load_prompts
from agentorg.utils.graph_state import MessageState
from agentorg.env.tools.RAG.retrievers.faiss_retriever import FaissRetrieverExecutor
from agentorg.env.tools.RAG.retrievers.milvus_retriever import MilvusRetrieverExecutor

logger = logging.getLogger(__name__)


class RetrieveEngine():
    @staticmethod
    def faiss_retrieve(state: MessageState):
        # get the input message
        user_message = state['user_message']

        # Search for the relevant documents
        prompts = load_prompts(state["bot_config"])
        docs = FaissRetrieverExecutor.load_docs(database_path=os.environ.get("DATA_DIR"))
        retrieved_text = docs.search(user_message.history, prompts["retrieve_contextualize_q_prompt"])

        state["message_flow"] = retrieved_text
        return state
    
    @staticmethod
    def milvus_retrieve(state: MessageState):
        # get the input message
        user_message = state['user_message']

        # Search for the relevant documents
        milvus_retriever = MilvusRetrieverExecutor(state["bot_config"])
        retrieved_text, retriever_params = milvus_retriever.retrieve(user_message.history)

        state["message_flow"] = retrieved_text
        state["metadata"]["tool_response"] = retriever_params
        return state
