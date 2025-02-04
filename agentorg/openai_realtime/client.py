import asyncio
import base64
from collections import defaultdict
import json
import os
from typing import List
import uuid
import numpy as np
import websockets
import logging

logger = logging.getLogger(__name__)

CHANGE_CONTEXT_FUNC_NAME = "change_context"

chng_ctx_func_def = {
    "type": "function",
    "name": CHANGE_CONTEXT_FUNC_NAME,
    "description": "Use this function when the user no longer wants to use the current tool and wants to talk/inquire about something else.",
}

class RealtimeClient:
    def __init__(self, telephony_mode: bool = False):
        self.ws = None
        self.modalities: List[str] = ["text"]
        self.prompt = ""
        self.turn_detection = {
            "type": "server_vad", 
            "create_response": False,
            "silence_duration_ms": 750,
        }
        self.internal_queue: asyncio.Queue = asyncio.Queue()
        self.external_queue: asyncio.Queue = asyncio.Queue()
        self.input_audio_buffer_event_queue: asyncio.Queue = asyncio.Queue()
        self.text_buffer = defaultdict(str)
        self.telephony_mode = telephony_mode
        self.input_audio_format = "g711_ulaw" if telephony_mode else "pcm16"
        self.output_audio_format = "g711_ulaw" if telephony_mode else "pcm16"

    def set_audio_modality(self) -> None:
        self.modalities = ["text", "audio"]

    def set_text_modality(self) -> None:
        self.modalities = ["text"]

    async def connect(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.ws = await websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
            extra_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        )

    async def close(self) -> None:
        await self.ws.close()

    def set_automatic_turn_detection(self) -> None:
        self.turn_detection = {"type": "server_vad", "create_response": False}

    async def update_session(self) -> None:
        event = {
            "type": "session.update",
            "session": {
                "turn_detection": self.turn_detection,
                "input_audio_format": self.input_audio_format,
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "output_audio_format": self.output_audio_format,
                "voice": 'alloy',
                "instructions": self.prompt,
                "modalities": self.modalities,
                "temperature": 0.8,
            }
        }
        logger.info(f"Updating session to {event}")
        await self.ws.send(json.dumps(event))

    async def send_audio(self, b64_encoded_audio: str) -> None:
        event = {
            "type": "input_audio_buffer.append",
            "audio": b64_encoded_audio
        }
        await self.ws.send(json.dumps(event))

    async def truncate_audio(self, item_id: str, audio_end_ms: int) -> None:
        logger.info(f"Truncating audio for item_id: {item_id} at {audio_end_ms} ms")
        event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms
        }
        await self.ws.send(json.dumps(event))

    async def commit_audio(self) -> None:
        event = {
            "type": "input_audio_buffer.commit"
        }
        await self.ws.send(json.dumps(event))

    async def create_response(self):
        logger.info("Creating response")
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def wait_till_input_audio(self) -> bool:
        logger.info("Waiting for input audio buffer speech stopped event")
        while True:
            openai_message = await self.input_audio_buffer_event_queue.get()
            if openai_message is None:
                return False
            # if openai_message.get("type") == "input_audio_buffer.speech_stopped":
            #     return True
            elif openai_message.get("type") == "input_audio_buffer.committed":
                return True
            else:
                logger.info(f"Skipping message(wait_till_input_audio): {openai_message}")

    async def add_function_call_output(self, call_id: str, output: str) -> None:
        await self.ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    },
                }
            )
        )

    async def execute_tool(self, openai_tool_def: dict, topic: str):
        while True:
            logger.info("In execute tool loop creating response...")
            msg_id = str(uuid.uuid4())

            event = {
                "type": "response.create",
                "response": {
                    # Set metadata to help identify responses sent back from the model
                    "metadata": { "topic": topic, "msg_id": msg_id },

                    "modalities": ["audio","text"],
                    "instructions": f"The user is interested in using the function: {openai_tool_def['name']} with the given parameters. If the parameters are missing work with the user to get the missing parameters. If the user reply is not related to the given tool or if the user asks something else use the function {chng_ctx_func_def['name']} to change the context.",
                    "tools": [
                        openai_tool_def,
                        chng_ctx_func_def
                    ],
                    "tool_choice": "required"
                }
            }
            await self.ws.send(json.dumps(event))

            while True:
                logger.info("In execute tool loop waiting for event..")
                openai_message = await self.internal_queue.get()
                if openai_message.get("type") == "response.done" and openai_message.get("response") and openai_message["response"].get("metadata") and openai_message["response"]["metadata"].get("msg_id") == msg_id:
                    logger.info(f"Received response.done for tool response: {openai_message}")
                    for output in openai_message["response"]["output"]:
                        if output.get("name") == openai_tool_def["name"] and "arguments" in output:
                            res = postprocess_json(output["arguments"])
                            res["call_id"] = output["call_id"]
                            await self.external_queue.put({
                                "type": "message",
                                "origin": "bot",
                                "id": msg_id,
                                "text": "Tool Call:\n\n" + str(openai_tool_def) + "\n\nResponse:\n\n" + str(res),
                                "audio_url": "",
                                "debug": True
                            })
                            logger.info("realtime execute_tool completed")
                            return res
                        if output.get("name") == chng_ctx_func_def["name"]:
                            logger.info("User wants to switch context")
                            await self.external_queue.put({
                                "type": "message",
                                "origin": "bot",
                                "id": msg_id,
                                "text": "User wants to change context",
                                "audio_url": "",
                                "debug": True
                            })
                            return {
                                "change_context": True
                            }
                    logger.error(f"Arguments not found in tool response: {openai_message}. Hence generating response again...")
                    await self.wait_till_input_audio()
                    break
                logger.info(f"Skipping message in execute tool: {openai_message}")

    async def get_text_response(self, text: str, topic: str) -> dict:
        for attempt in range(3):
            logger.info(f"create text response: {topic}: {text}")

            msg_id = str(uuid.uuid4())
            await self.ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    # Setting to "none" indicates the response is out of band,
                    # and will not be added to the default conversation
                    "conversation": "none",

                    # Set metadata to help identify responses sent back from the model
                    "metadata": { "topic": topic, "msg_id": msg_id },

                    "modalities": ["text"],
                    "instructions": text
                }
            }))

            await self.external_queue.put({
                "type": "message",
                "origin": "bot",
                "id": msg_id,
                "text":  "Prompt:\n\n" + text,
                "audio_url": "",
                "debug": True
            })

            while True:
                logger.info(f"Waiting for text response, attempt: {attempt+1}")
                openai_message = await self.internal_queue.get()

                if openai_message.get("type") == "response.done" and openai_message.get("response") and openai_message["response"]["status"] == "failed" and openai_message["response"].get("metadata") and openai_message["response"]["metadata"].get("topic") == topic:
                    logger.error(f"Failed to get response(openai error): {openai_message}")
                    break
                if openai_message.get("type") == "response.done" and openai_message.get("response") and openai_message["response"].get("metadata") and openai_message["response"]["metadata"].get("topic") == topic:
                    # res = postprocess_json(openai_message.response.output[0].content[0].text)
                    logger.info(f"Received text response: {openai_message}")
                    try:
                        await self.external_queue.put({
                            "type": "message",
                            "origin": "bot",
                            "id": msg_id,
                            "text": "Prompt:\n\n" + text + "\n\nResponse:\n\n" + openai_message["response"]["output"][0]["content"][0]["text"],
                            "audio_url": "",
                            "debug": True
                        })
                        res = postprocess_json(openai_message["response"]["output"][0]["content"][0]["text"])
                    except Exception as e:
                        logger.error(f"Error processing response: {e}, openai message: {openai_message},retrying...")
                        event = {
                            "type": "message",
                            "origin": "bot",
                            "id": msg_id,
                            "text": f"Wrong response format for prompt:\n\n{text}\n\nRetrying... Recieved Message: " + str(openai_message["response"]["output"][0]["content"]),
                            "audio_url": "",
                            "debug": True
                        }
                        await self.external_queue.put(event)
                        break
                    logger.info(f"Processed response: {res}")
                    return res
                else:
                    logger.info(f"Skipping message: {openai_message}")

    async def create_audio_response(self, prompt: str):
        logger.info(f"Creating audio response with: {prompt}")
        self.prompt = prompt
        self.set_audio_modality()
        await self.update_session()
        await self.create_response()

    async def receive_events(self) -> None:
        async for openai_message in self.ws:
            openai_event = json.loads(openai_message)
            event_type = openai_event.get("type")
            logger.info(f"Received event type: {event_type}")

            if event_type == 'error':
                logger.error(f"Error from OpenAI: {openai_event}")
                continue

            if event_type == 'response.done':
                await self.internal_queue.put(openai_event)

            if event_type == 'response.text.done' and 'text' in openai_event:
                await self.internal_queue.put(openai_event)

            if event_type == 'response.audio.delta' and 'delta' in openai_event:
                event = {
                    "type": "audio_stream",
                    "origin": "bot",
                    "id": openai_event['item_id'],
                    "audio_bytes": base64.b64encode(base64.b64decode(openai_event['delta'])).decode('utf-8') if self.telephony_mode else np.frombuffer(base64.b64decode(openai_event['delta']), np.int16).tolist(),
                }
                await self.external_queue.put(event)

            if event_type == 'response.audio_transcript.delta' and 'delta' in openai_event:
                self.text_buffer[openai_event['item_id']] += openai_event['delta']
                event = {
                    "type": "text_stream",
                    "origin": "bot",
                    "id": openai_event['item_id'],
                    "text": self.text_buffer[openai_event['item_id']]
                }
                await self.external_queue.put(event)

            if event_type == 'response.audio_transcript.done':
                event = {
                    "type": "message",
                    "origin": "bot",
                    "id": openai_event['item_id'],
                    "text": openai_event['transcript'],
                    "audio_url": "",
                }
                await self.external_queue.put(event)

            if event_type == 'input_audio_buffer.speech_started':
                await self.input_audio_buffer_event_queue.put(openai_event)
                await self.external_queue.put({"type": "input_audio_buffer.speech_started"})

            if event_type == 'input_audio_buffer.speech_stopped':
                await self.input_audio_buffer_event_queue.put(openai_event)
                await self.external_queue.put({"type": "input_audio_buffer.speech_stopped"})

            if event_type == 'input_audio_buffer.committed':
                await self.input_audio_buffer_event_queue.put(openai_event)

            if event_type == 'conversation.item.created' and openai_event.get('item') and openai_event['item'].get('role') and (openai_event["item"]["role"] == "user" or openai_event["item"]["role"] == "assistant"):
                event = {
                    "type": "message",
                    "origin": "user" if openai_event["item"]["role"] == "user" else "bot",
                    "id": openai_event['item']['id'],
                    "text": " ",
                    "audio_url": "",
                }
                await self.external_queue.put(event)

            if event_type == 'conversation.item.input_audio_transcription.completed':
                event = {
                    "type": "message",
                    "origin": "user",
                    "id": openai_event['item_id'],
                    "text": openai_event['transcript'],
                    "audio_url": "",
                }
                await self.external_queue.put(event)

            if event_type == 'response.function_call_arguments.done':
                await self.internal_queue.put(openai_event)

        logger.info("receive_events ended")
        await self.end_queues()
        await self.close()

    async def end_queues(self):
        await self.internal_queue.put(None)
        await self.input_audio_buffer_event_queue.put(None)
        await self.external_queue.put(None)

def postprocess_json(input: str)-> dict:
    input = input.replace("'", '"')
    valid_phrases = ['"', '{', '}', '[', ']']

    valid_lines = []
    for line in input.split('\n'):
        if len(line) == 0:
            continue
        # If the line not starts with any of the valid phrases, skip it
        should_skip = not any([line.strip().startswith(phrase) for phrase in valid_phrases])
        if should_skip:
            continue
        valid_lines.append(line)

    generated_result = "\n".join(valid_lines)
    result = json.loads(generated_result)
    if len(result.keys()) == 0:
        raise Exception(f"Failed to parse response: {input}")
    return result
