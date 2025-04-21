# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import hashlib
import os
import threading
import time
import uuid
from typing import List

import uvicorn
from fastapi import FastAPI, File, Form, Body, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydub import AudioSegment
from starlette.middleware.cors import CORSMiddleware
from whisper_model import WhisperModel
from faster_whisper_model import FasterWhisperModel, BYTES_PER_SAMPLE, SAMPLE_RATE

from comps import CustomLogger
from comps.cores.proto.api_protocol import AudioTranscriptionResponse, RealtimeTranscriptionSession

logger = CustomLogger("whisper")
logflag = os.getenv("LOGFLAG", False)

# audio configuration constantsgit 
DEFAULT_CHUNK_DURATION_MS = 3000  # default chunk size is 2.5 seconds
DEFAULT_FRAMES_PER_CHUNK = SAMPLE_RATE  # 16000 frames per second
DEFAULT_DATA_TIMEOUT_MS = 1500  # set 1500ms for timeout

app = FastAPI()
asr = None
streaming_asr = None
streaming_asr_ready = threading.Event()


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


def generate_session_id(curr_time):
    """
    generate session id with prefix 'sess_' and current time hash

    Returns:
        session id
    """
    if curr_time is None:
        curr_time = time.time()
    # change timestamp to 'utf-8' encoded bytes
    time_str = f"{curr_time}"
    time_bytes = time_str.encode('utf-8')

    # calculate MD5 hash
    md5_hash = hashlib.md5(time_bytes).hexdigest()

    # return session id
    return f"sess_{md5_hash}"


def is_buffer_ready(buffer: bytes, chunk_size: int) -> bool:
    """check if the buffer is ready for a complete chunk"""
    return len(buffer) >= chunk_size


def init_streaming_whisper(model_size: str, device: str, compute_type: str, language: str):
    """init streaming whisper service in a thread"""
    global streaming_asr
    if device == "cpu":
        """initialize the faster-whisper based streaming whisper model in a separate thread"""
        try:
            streaming_asr = FasterWhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type
            )
            streaming_asr_ready.set()
            logger.info("Faster Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Faster Whisper model: {e}")
            streaming_asr_ready.set()  # set the event even if it fails to avoid permanent waiting
    else:
        """initialize the streaming whisper model on other platforms in a separate thread"""
        try:
            streaming_asr = WhisperModel(
                model_name_or_path=model_size,
                language=language,
                device=device
            )
            streaming_asr_ready.set()
            logger.info("Streaming Whisper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize streaming Whisper model: {e}")
            streaming_asr_ready.set()


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/asr")
async def audio_to_text(request: Request):
    logger.info("Whisper generation begin.")
    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    request_dict = await request.json()
    audio_b64_str = request_dict.pop("audio")
    with open(file_name, "wb") as f:
        f.write(base64.b64decode(audio_b64_str))

    audio = AudioSegment.from_file(file_name)
    audio = audio.set_frame_rate(16000)

    audio.export(f"{file_name}", format="wav")
    try:
        asr_result = asr.audio2text(file_name)
    except Exception as e:
        logger.error(e)
        asr_result = e
    finally:
        os.remove(file_name)
    return {"asr_result": asr_result}


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),  # Handling the uploaded file directly
    model: str = Form("openai/whisper-small"),
    language: str = Form("english"),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    timestamp_granularities: List[str] = Form(None),
):
    logger.info("Whisper generation begin.")
    audio_content = await file.read()
    # validate the request parameters
    if model != asr.asr_model_name_or_path:
        raise Exception(
            f"ASR model mismatch! Please make sure you pass --model_name_or_path or set environment variable ASR_MODEL_PATH to {model}"
        )
    asr.language = language
    if prompt is not None or response_format != "json" or temperature != 0 or timestamp_granularities is not None:
        logger.warning(
            "Currently parameters 'language', 'response_format', 'temperature', 'timestamp_granularities' are not supported!"
        )

    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    # Save the uploaded file
    with open(file_name, "wb") as buffer:
        buffer.write(audio_content)

    audio = AudioSegment.from_file(file_name)
    audio = audio.set_frame_rate(16000)

    audio.export(f"{file_name}", format="wav")

    try:
        asr_result = asr.audio2text(file_name)
    except Exception as e:
        logger.error(e)
        asr_result = e
    finally:
        os.remove(file_name)

    return AudioTranscriptionResponse(text=asr_result)


@app.post("/v1/realtime/transcription_sessions")
async def create_realtime_transcription_session(
    input_audio_format: str = Body("pcm16"),
    input_audio_noise_reduction: dict = Body(None),
    input_audio_transcription: dict = Body("json"),
    modalities: List = Body(None),
    turn_detection: dict = Body(None),
    include: str = Body(None)
):
    logger.info("Creating realtime transcription session.")
    curr_time = time.time()
    session_id = generate_session_id(curr_time)
    expire_time = int(curr_time + 300)  # set the expire time as 5 mins later, now mimicing the OpenAI behavior
    if modalities and len(modalities) != 0:
        if modalities[0] != "text":
            logger.info("Do not support modalities other than text for now.")
            modalities = "text"
    else:
        modalities = ["text"]

    # Initialize configuration dictionary
    if streaming_asr is None:
        streaming_asr_ready.wait(timeout=15)
    language = "en"
    if hasattr(streaming_asr, "language"):
        language = streaming_asr.language
    valid_config = {
        "model": streaming_asr.model_size_or_path,
        "language": language,
        "prompt": ""
    }

    if isinstance(input_audio_transcription, dict):
        # Check for unknown keys
        unknown_keys = [
            k for k in input_audio_transcription.keys() if k not in valid_config]
        if unknown_keys:
            logger.warning(
                f"Found unknown configuration keys: {unknown_keys}, these will be ignored")

        # Handle model configuration
        if "model" in input_audio_transcription:
            model = input_audio_transcription["model"]
            if model != valid_config["model"]:
                logger.warning(
                    f"Unmatched model: {model}. Now is using model {streaming_asr.model_size_or_path}")

        # Handle language configuration
        if "language" in input_audio_transcription and valid_config["language"] != input_audio_transcription["language"]:
            logger.warning(
                f"Unmatched language setting. Now is using language {streaming_asr.language}")

    # Check if noise reduction is requested
    if input_audio_noise_reduction:
        logger.warning("Audio noise reduction is not supported at this time")

    if turn_detection:
        logger.warning("Audio turn detection is only supported in cpu mode")

    if include:
        logger.warning("Audio include setting is not supported at this time")

    return RealtimeTranscriptionSession(id=session_id, expires_at=expire_time,
                                        input_audio_format=input_audio_format, input_audio_transcription=valid_config,
                                        modalities=modalities, turn_detection=None)


@app.websocket("/v1/realtime")
async def audio_transcriptions_streaming(websocket: WebSocket, intent: str = "transcription"):
    """
    This endpoint is used to stream the transcription of the audio input.
    Args:
        websocket: WebSocket
        intent: String, default is "transcription"
    """
    if intent != "transcription":
        logger.warning(
            "Unsupported function. Currently only support the 'transcription' intent.")
        await websocket.close()
        return

    await websocket.accept()

    chunk_size = int((DEFAULT_CHUNK_DURATION_MS *
                     DEFAULT_FRAMES_PER_CHUNK) / 1000) * BYTES_PER_SAMPLE
    logger.info(f"Chunk size: {chunk_size}")

    # wait for the faster-whisper model to be initialized
    if not streaming_asr_ready.wait(timeout=10):
        await websocket.send_json({
            "event_id": "event_0",
            "type": "error",
            "error": {
                "type": "initialization_timeout",
                "code": "initialization_timeout",
                "message": "The streaming asr failed to initialize.",
                "param": None,
                "event_id": "event_0"
            }
        })
        return

    if streaming_asr is None:
        await websocket.send_json({
            "event_id": "event_0",
            "type": "error",
            "error": {
                "type": "initialization_failure",
                "code": "initialization_failure",
                "message": "The streaming asr model failed to initialize.",
                "param": None,
                "event_id": "event_0"
            }
        })
        return

    try:
        # initialize audio buffer
        audio_buffer = bytearray()
        idx_list = {}
        item_id = 0

        while True:
            try:
                # receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=DEFAULT_DATA_TIMEOUT_MS / 1000
                    )
                except asyncio.TimeoutError:
                    if len(audio_buffer) > 0:
                        logger.info(
                            f"Data receive timeout ({DEFAULT_DATA_TIMEOUT_MS}ms), processing final {len(audio_buffer)} bytes of audio data")
                        await streaming_asr.audio2text_streaming(websocket=websocket, audio_data=bytes(audio_buffer), item_id=item_id+1, event_id=event_id, is_final=True)
                    break

                # process the audio data
                if message.get("type") == "input_audio_buffer.append":
                    # check if event_id exists in the index map
                    event_id = message.get("event_id")
                    if event_id in idx_list:
                        item_id = idx_list[event_id] + 1
                    else:
                        idx_list[event_id] = 0
                        item_id = 0

                    # fetch audio data
                    audio_data = base64.b64decode(message.get("audio", ""))
                    audio_buffer.extend(audio_data)

                    # process complete chunks
                    if is_buffer_ready(audio_buffer, chunk_size):
                        await streaming_asr.audio2text_streaming(websocket=websocket, audio_data=bytes(audio_buffer[:chunk_size]), item_id=item_id, event_id=event_id)
                        audio_buffer = audio_buffer[chunk_size:]

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed unexpectedly")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
                await websocket.send_json({
                    "status": "error",
                    "error": str(e)
                })
                break

    except Exception as e:
        logger.error(f"Error in audio streaming: {e}")
        await websocket.send_json({
            "status": "error",
            "error": str(e)
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7066)
    parser.add_argument("--model_name_or_path", type=str,
                        default="openai/whisper-small")
    parser.add_argument("--streaming_model_name_or_path",
                        type=str, default="tiny")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--return_timestamps", type=str, default=True)
    parser.add_argument("--compute-type", type=str, default="int8")

    args = parser.parse_args()
    asr = WhisperModel(
        model_name_or_path=args.model_name_or_path,
        language=args.language,
        device=args.device,
        return_timestamps=args.return_timestamps,
    )

    streaming_whisper_thread = threading.Thread(
        target=init_streaming_whisper,
        args=(args.streaming_model_name_or_path,
              args.device, args.compute_type, args.language),
        daemon=True
    )
    streaming_whisper_thread.start()

    uvicorn.run(app, host=args.host, port=args.port)
