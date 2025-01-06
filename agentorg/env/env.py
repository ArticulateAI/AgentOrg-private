import logging
import uuid

from agentorg.env.workers.worker import WORKER_REGISTRY
from agentorg.env.tools.tools import Tool, TOOL_REGISTRY
from agentorg.env.planner.function_calling import FunctionCallingPlanner
from agentorg.utils.graph_state import StatusEnum
from agentorg.orchestrator.NLU.nlu import SlotFilling


logger = logging.getLogger(__name__)

class Env():
    def __init__(self, tools, workers, slotsfillapi):
        self.tools = self.initialize_tools(tools)
        self.workers = self.initialize_workers(workers)
        self.slotfillapi = self.initialize_slotfillapi(slotsfillapi)
        self.planner = FunctionCallingPlanner(
            tools_map=self.tools
        )

    def initialize_tools(self, tools):
        # Create tool_regrstry from the tools list
        tool_registry = {}
        for tool in tools:
            tool_registry[tool["name"]] = {"execute": TOOL_REGISTRY[tool["name"]], "fixed_args": tool.get("fixed_args", {})}
        return tool_registry

    def initialize_workers(self, workers):
        # Create worker_registry from the workers list
        worker_registry = {}
        for worker in workers:
            worker_registry[worker] = WORKER_REGISTRY[worker]
        return worker_registry
    
    def initialize_slotfillapi(self, slotsfillapi):
        return SlotFilling(slotsfillapi)

    def step(self, name, message_state, params):
        if name in self.tools:
            logger.info(f"{name} tool selected")
            tool: Tool = self.tools[name]["execute"]()
            tool.init_slotfilling(self.slotfillapi)
            response_state = tool.execute(message_state, **self.tools[name]["fixed_args"])
            params["history"] = response_state.get("trajectory", [])
            current_node = params.get("curr_node")
            params["node_status"][current_node] = response_state.get("status", StatusEnum.COMPLETE.value)
                
        elif name in self.workers:
            logger.info(f"{name} worker selected")
            worker = self.workers[name]()
            response_state = worker.execute(message_state)
            call_id = str(uuid.uuid4())
            params["history"].append({'content': None, 'role': 'assistant', 'tool_calls': [{'function': {'arguments': "", 'name': name}, 'id': call_id, 'type': 'function'}], 'function_call': None})
            params["history"].append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": response_state["response"]
            })
        else:
            logger.info("planner selected")
            action, response_state, msg_history = self.planner.execute(message_state, params["history"])
        
        logger.info(f"Response state from {name}: {response_state}")
        return response_state, params
