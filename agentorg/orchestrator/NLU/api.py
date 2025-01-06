import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import logging
import string

from openai import OpenAI
from fastapi import FastAPI, Response

from agentorg.utils.graph_state import Slots, Slot
from dotenv import load_dotenv
load_dotenv()

from agentorg.utils.model_config import MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_NLU = """According to the conversation, decide what is the user's intent in the last turn? \nHere are the definitions for each intent:\n{definition}\nHere are some sample utterances from user that indicate each intent:\n{exemplars}\nConversation:\n{formatted_chat}\n\nOnly choose from the following options.\n{intents_choice}\n\nAnswer:
"""


class OpenAIAPI:
    def __init__(self):
        self.client = OpenAI()


class NLUOpenAIAPI(OpenAIAPI):
    def __init__(self):
        super().__init__()
        self.user_prefix = "user"
        self.assistant_prefix = "assistant"

    def get_response(self, sys_prompt, response_format="text", debug_text="none", params=MODEL):
        logger.info(f"gpt system_prompt for {debug_text} is \n{sys_prompt}")
        dialog_history = {"role": "system", "content": sys_prompt}
        completion = self.client.chat.completions.create(
            model=params.get("model_type_or_path", "gpt-4o"),
            response_format={"type": "json_object"} if response_format=="json" else {"type": "text"},
            messages=[dialog_history],
            n=1,
            temperature = 0.7
        )
        response = completion.choices[0].message.content
        logger.info(f"response for {debug_text} is \n{response}")
        return response

    def format_input(self, intents, chat_history_str) -> str:
        """Format input text before feeding it to the model."""
        intents_choice, definition_str, exemplars_str = "", "", ""
        idx2intents_mapping = {}
        multiple_choice_index = dict(enumerate(string.ascii_lowercase))
        count = 0
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

                    count += 1

        system_prompt = SYSTEM_PROMPT_NLU.format(
            definition=definition_str,
            exemplars=exemplars_str,
            intents_choice=intents_choice,
            formatted_chat=chat_history_str,
        )
        return system_prompt, idx2intents_mapping

    def predict(
        self,
        text,
        intents,
        chat_history_str
    ) -> str:

        system_prompt, idx2intents_mapping = self.format_input(
            intents, chat_history_str
        )
        response = self.get_response(
            system_prompt, debug_text="get intent"
        )
        logger.info(f"postprocessed intent response: {response}")
        try:
            pred_intent_idx = response.split(")")[0]
            pred_intent = idx2intents_mapping[pred_intent_idx]
        except:
            pred_intent = response.strip().lower()
        logger.info(f"postprocessed intent response: {pred_intent}")
        return pred_intent


class SlotFillOpenAIAPI(OpenAIAPI):
    def __init__(self):
        super().__init__()
        self.user_prefix = "user"
        self.assistant_prefix = "assistant"

    def get_response(self, sys_prompt, debug_text="none", params=MODEL):
        logger.info(f"gpt system_prompt for {debug_text} is \n{sys_prompt}")
        dialog_history = {"role": "system", "content": sys_prompt}
        completion = self.client.beta.chat.completions.parse(
            model=params.get("model_type_or_path", "gpt-4o"),
            messages=[dialog_history],
            response_format=Slots,
            n=1,
            temperature = 0.7
        )
        response = completion.choices[0].message
        if (response.refusal):
            return None
        logger.info(f"response for {debug_text} is \n{response.parsed}")
        return response.parsed

    def format_input(self, slots: Slots, chat_history_str) -> str:
        """Format input text before feeding it to the model."""
        system_prompt = f"Given the conversation and definition of dialog states definition, update the value of following dialogue states.\nDialogue Statues:\n{slots}\nConversation:\n{chat_history_str}\n\n"
        return system_prompt

    def predict(
        self,
        slots,
        chat_history_str
    ):
        slots = [Slot(**slot_data) for slot_data in slots]
        system_prompt = self.format_input(
            slots, chat_history_str
        )
        response = self.get_response(
            system_prompt, debug_text="get slots"
        )
        if not response:
            logger.info(f"Failed to update dialogue states")
            return slots
        logger.info(f"Updated dialogue states: {response}")
        return response


app = FastAPI()
nlu_openai = NLUOpenAIAPI()
slotfilling_openai = SlotFillOpenAIAPI()


@app.post("/nlu/predict")
def predict(data: dict, res: Response):
    logger.info(f"Received data: {data}")
    pred_intent = nlu_openai.predict(**data)

    logger.info(f"pred_intent: {pred_intent}")
    return {"intent": pred_intent}

@app.post("/slotfill/predict")
def predict(data: dict, res: Response):
    logger.info(f"Received data: {data}")
    results = slotfilling_openai.predict(**data)

    logger.info(f"pred_slots: {results.slots}")
    return results.slots