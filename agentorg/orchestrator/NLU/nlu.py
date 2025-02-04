import string
from typing import List, Tuple
import requests
import logging
from dotenv import load_dotenv

import langsmith as ls

from agentorg.openai_realtime.client import CHANGE_CONTEXT_FUNC_NAME, RealtimeClient
from agentorg.utils.trace import TraceRunName
from agentorg.utils.graph_state import Slots, Slot
from agentorg.orchestrator.NLU.api import nlu_openai, slotfilling_openai

load_dotenv()
logger = logging.getLogger(__name__)


class NLU:
    def __init__(self, url=None, realtime_client: RealtimeClient = None):
        self.url = url
        self.realtime_client = realtime_client

    def format_intent_for_realtime(self, intents) -> str:
        """Format input text before feeding it to the model."""
        intents_choice, definition_str, exemplars_str = "", "", ""
        idx2intents_mapping = {}
        multiple_choice_index = dict(enumerate(string.ascii_lowercase))
        count = 0
        ex_intent = ""
        ex_intent_idx = ""
        for intent_k, intent_v in intents.items():
            if len(intent_v) == 1:
                intent_name = intent_k
                idx2intents_mapping[multiple_choice_index[count]] = intent_name
                definition = intent_v[0].get("attribute", {}).get("definition", "")
                sample_utterances = intent_v[0].get("attribute", {}).get("sample_utterances", [])

                if definition:
                    definition_str += (
                        f"{multiple_choice_index[count]}) {intent_name}: {definition}\n"
                    )
                if sample_utterances:
                    exemplars = "\n".join(sample_utterances)
                    exemplars_str += (
                        f"{multiple_choice_index[count]}) {intent_name}: \n{exemplars}\n"
                    )
                intents_choice += f"{multiple_choice_index[count]}) {intent_name}\n"
                ex_intent = intent_name
                ex_intent_idx = multiple_choice_index[count]
                count += 1

            else:
                for idx, intent in enumerate(intent_v):
                    intent_name = f'{intent_k}__<{idx}>'
                    idx2intents_mapping[multiple_choice_index[count]] = intent_name
                    definition = intent.get("attribute", {}).get("definition", "")
                    sample_utterances = intent.get("attribute", {}).get("sample_utterances", [])

                    if definition:
                        definition_str += (
                            f"{multiple_choice_index[count]}) {intent_name}: {definition}\n"
                        )
                    if sample_utterances:
                        exemplars = "\n".join(sample_utterances)
                        exemplars_str += (
                            f"{multiple_choice_index[count]}) {intent_name}: \n{exemplars}\n"
                        )
                    intents_choice += f"{multiple_choice_index[count]}) {intent_name}\n"
                    ex_intent = intent_name
                    ex_intent_idx = multiple_choice_index[count]
                    count += 1

        system_prompt = f"""Analyze the conversation so far. According to user's last turn, what is the user's intention? Only choose from the following options: {intents_choice}. Output the response in JSON format. Example: {{'intent': '{ex_intent}', 'intent_idx': '{ex_intent_idx}'}}\nThe JSON keys 'intent' should be the intent name and 'intent_idx' should be the corresponding index of the intent in the list of intents. ONLY OUTPUT JSON FORMAT with the keys 'intent' and 'intent_idx'!"""

        return system_prompt, idx2intents_mapping
    
    async def realtime_intent_detetion(self, intents):
        system_prompt, idx2intents_mapping = self.format_intent_for_realtime(intents)
        response =  await self.realtime_client.get_text_response(system_prompt, "intent_check")
        logger.info(f"realtime_intent_detetion response: {response}")
        try:
            intentx_idx = response["intent_idx"]
            pred_intent = idx2intents_mapping[intentx_idx]
        except Exception as e:
            pred_intent = response["intent"].strip().lower()
        logger.info(f"realtime_intent_detetion postprocessed: {pred_intent}")
        return pred_intent

    def execute(self, text:str, intents:dict, chat_history_str:str, metadata:dict) -> str:
        logger.info(f"candidates intents of NLU: {intents}")
        data = {
            "text": text,
            "intents": intents,
            "chat_history_str": chat_history_str
        }
        if self.url:
            logger.info(f"Using NLU API to predict the intent")
            response = requests.post(self.url, json=data)
            if response.status_code == 200:
                results = response.json()
                pred_intent = results['intent']
                logger.info(f"pred_intent is {pred_intent}")
            else:
                pred_intent = "others"
                logger.error('Remote Server Error when predicting NLU')
        else:
            logger.info(f"Using NLU function to predict the intent")
            pred_intent = nlu_openai.predict(**data)
            logger.info(f"pred_intent is {pred_intent}")

        with ls.trace(name=TraceRunName.NLU, inputs=data) as rt:
            rt.end(
                outputs=pred_intent,
                metadata={"chat_id": metadata.get("chat_id"), "turn_id": metadata.get("turn_id")}
            )
        return pred_intent
    

class SlotFilling:
    def __init__(self, url=None, realtime_client: RealtimeClient = None):
        self.url = url
        self.realtime_client = realtime_client

    async def realtime_execute(self, slots: List[Slot], openai_tool_def: dict) -> Tuple[List[Slot], bool]:
        params = await self.realtime_client.execute_tool(openai_tool_def, "slot_fill")
        if CHANGE_CONTEXT_FUNC_NAME in params:
            return [], True
        for slot in slots:
            if slot.name in params:
                slot.value = params[slot.name]
        return slots, False


    def execute(self, slots:list, chat_history_str:str, metadata: dict) -> dict:
        logger.info(f"extracted slots: {slots}")
        if not slots: return []
        
        data = {
            "slots": [slot.dict() for slot in slots],
            "chat_history_str": chat_history_str
        }
        if self.url:
            logger.info(f"Using Slot Filling API to predict the slots")
            response = requests.post(self.url, json=data)
            if response.status_code == 200:
                pred_slots = response.json()
                logger.info(f"The raw pred_slots is {pred_slots}")
                pred_slots = [Slot(**pred_slot) for pred_slot in pred_slots]
                logger.info(f"pred_slots is {pred_slots}")
            else:
                pred_slots = slots
                logger.error('Remote Server Error when predicting Slot Filling')
        else:
            logger.info(f"Using Slot Filling function to predict the slots")
            pred_slots = slotfilling_openai.predict(**data).slots
            # pred_slots = [Slot(**pred_slot) for pred_slot in pred_slots]
            logger.info(f"pred_slots is {pred_slots}")
        with ls.trace(name=TraceRunName.SlotFilling, inputs=data) as rt:
            rt.end(
                outputs=pred_slots,
                metadata={"chat_id": metadata.get("chat_id"), "turn_id": metadata.get("turn_id")}
            )
        return pred_slots
