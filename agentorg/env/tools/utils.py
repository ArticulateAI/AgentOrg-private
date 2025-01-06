import logging

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from agentorg.env.prompts import load_prompts
from agentorg.utils.utils import chunk_string
from agentorg.utils.graph_state import MessageState
from agentorg.utils.model_config import MODEL


logger = logging.getLogger(__name__)
    

class ToolGenerator():
    @staticmethod
    def generate(state: MessageState):
        user_message = state['user_message']
        
        prompts = load_prompts(state["bot_config"])
        llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
        prompt = PromptTemplate.from_template(prompts["generator_prompt"])
        input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "formatted_chat": user_message.history})
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = llm | StrOutputParser()
        answer = final_chain.invoke(chunked_prompt)

        state["response"] = answer
        return state

    @staticmethod
    def context_generate(state: MessageState):
        llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
        # get the input message
        user_message = state['user_message']
        message_flow = state['message_flow']
        logger.info(f"Retrieved texts (from retriever/search engine to generator): {message_flow[:50]} ...")
        
        # generate answer based on the retrieved texts
        prompts = load_prompts(state["bot_config"])
        prompt = PromptTemplate.from_template(prompts["context_generator_prompt"])
        input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "formatted_chat": user_message.history, "context": message_flow})
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = llm | StrOutputParser()
        logger.info(f"Prompt: {input_prompt.text}")
        answer = final_chain.invoke(chunked_prompt)
        state["message_flow"] = ""
        state["response"] = answer

        return state