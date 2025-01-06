import logging
import os

from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI

from agentorg.env.workers.worker import BaseWorker, register_worker
from agentorg.utils.graph_state import MessageState
from agentorg.env.tools.utils import ToolGenerator
from agentorg.env.tools.RAG.retriever import RetrieveEngine
from agentorg.utils.model_config import MODEL


logger = logging.getLogger(__name__)


@register_worker
class RAGWorker(BaseWorker):

    description = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(self):
        super().__init__()
        self.action_graph = self._create_action_graph()
        self.llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)

    def choose_retriever(self, state: MessageState):
        if os.getenv("MILVUS_URI", ""):
            logger.info("Using Milvus retriever")
            return "milvus_retriever"
        logger.info("Using Faiss retriever")
        return "faiss_retriever"

    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("faiss_retriever", RetrieveEngine.faiss_retrieve)
        workflow.add_node("milvus_retriever", RetrieveEngine.milvus_retrieve)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        # Add edges
        workflow.add_conditional_edges(START, self.choose_retriever)
        workflow.add_edge("faiss_retriever", "tool_generator")
        workflow.add_edge("milvus_retriever", "tool_generator")
        return workflow

    def execute(self, msg_state: MessageState):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result
