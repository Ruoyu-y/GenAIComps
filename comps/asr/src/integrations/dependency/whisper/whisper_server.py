# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import os
import uuid
from typing import List
import wave
import uvicorn
import threading
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydub import AudioSegment
from starlette.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel as FasterWhisperModel
from whisper_model import WhisperModel
import asyncio

from comps import CustomLogger
from comps.cores.proto.api_protocol import AudioTranscriptionResponse

logger = CustomLogger("whisper")
logflag = os.getenv("LOGFLAG", False)

# audio configuration constants
SAMPLE_RATE = 16000  # Hz
BYTES_PER_SAMPLE = 2  # 16-bit audio
DEFAULT_CHUNK_DURATION_MS = 1000  # default chunk size is 1 second
DEFAULT_FRAMES_PER_CHUNK = SAMPLE_RATE  # 16000 frames per second

app = FastAPI()
asr = None
faster_asr = None
faster_asr_ready = threading.Event()


def init_faster_whisper(model_size: str, device: str, compute_type: str):
    """initialize the faster-whisper model in a separate thread"""
    global faster_asr
    try:
        faster_asr = FasterWhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type
        )
        faster_asr_ready.set()
        logger.info("Faster Whisper model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Faster Whisper model: {e}")
        faster_asr_ready.set()  # set the event even if it fails to avoid permanent waiting


def is_buffer_ready(buffer: bytes, chunk_size: int) -> bool:
    """check if the buffer is ready for a complete chunk"""
    return len(buffer) >= chunk_size


app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


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
            f"ASR model mismatch! Please make sure you pass --model_name_or_path \
            or set environment variable ASR_MODEL_PATH to {model}"
        )
    asr.language = language
    if prompt is not None or response_format != "json" or \
            temperature != 0 or timestamp_granularities is not None:
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


@app.websocket("/v1/asr/streaming")
async def audio_streaming(
    websocket: WebSocket,
    chunk_duration_ms: int = DEFAULT_CHUNK_DURATION_MS,
    # duration of each audio chunk in milliseconds
    language: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
    min_silence_duration_ms: int = 500,
    data_timeout_ms: int = 1500,  # timeout duration for receiving new data
    response_format: str = "json"  # 'json' or 'verbose_json'
):
    """
    WebSocket endpoint for streaming audio processing using faster-whisper.
    Args:
        chunk_duration_ms: Duration of each audio chunk in milliseconds
        language: Language code for transcription (e.g., 'en', 'zh', 'auto')
        beam_size: Beam size for decoding
        vad_filter: Whether to use voice activity detection
        min_silence_duration_ms: Minimum silence duration for VAD in milliseconds
        data_timeout_ms: Timeout duration for receiving new data in milliseconds
        response_format: Response format, either 'json' or 'verbose_json'
    """
    await websocket.accept()

    # calculate the size of audio chunk
    chunk_size = int(
        (chunk_duration_ms * DEFAULT_FRAMES_PER_CHUNK) / 1000) * BYTES_PER_SAMPLE
    logger.info(
        f"Audio chunk size: {chunk_size} bytes ({chunk_duration_ms}ms)")

    # wait for the faster-whisper model to be initialized
    if not faster_asr_ready.wait(timeout=10):  # timeout is 10 seconds
        await websocket.send_json({
            "status": "error",
            "error": "Faster Whisper model initialization timeout"
        })
        return

    if faster_asr is None:
        await websocket.send_json({
            "status": "error",
            "error": "Faster Whisper model initialization failed"
        })
        return

    # store all segments for final response
    all_segments = []
    total_duration = 0
    current_segment_id = 0

    async def process_audio_chunk(audio_data: bytes, is_final: bool = False):
        """process the audio data chunk"""
        if not audio_data:
            return

        # generate a temporary file name
        uid = str(uuid.uuid4())
        temp_filename = f"{uid}.wav"

        try:
            # create a WAV file
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # single channel
                wav_file.setsampwidth(BYTES_PER_SAMPLE)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_data)

            # use faster-whisper for speech recognition
            segments, info = faster_asr.transcribe(
                temp_filename,
                language=None if language == "auto" else language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=min_silence_duration_ms)
            )

            # process the recognition results
            transcription = []
            nonlocal current_segment_id, total_duration

            for segment in segments:
                # create segment data in OpenAI format
                segment_data = {
                    "id": current_segment_id,
                    "seek": 0,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": 0.0,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob
                }

                if response_format == "verbose_json":
                    transcription.append(segment_data)
                else:
                    # simplified format for normal json response
                    transcription.append({
                        "text": segment.text
                    })

                all_segments.append(segment_data)
                current_segment_id += 1
                total_duration = max(total_duration, segment.end)

            # prepare response based on format
            response = {
                "status": "success",
                "language": info.language,
                "language_probability": info.language_probability
            }

            if response_format == "verbose_json":
                response.update({
                    "task": "transcribe",
                    "duration": total_duration,
                    "segments": transcription,
                    "text": " ".join(s["text"].strip() for s in transcription)
                })
            else:
                response.update({
                    "segments": transcription,
                    "chunk_info": {
                        "duration_ms": chunk_duration_ms,
                        "bytes": len(audio_data),
                        "is_final": is_final
                    }
                })

            # send the recognition results
            await websocket.send_json(response)

            # if it is the last chunk of data, send the completion signal
            if is_final:
                if response_format == "verbose_json":
                    # Send a complete transcription in OpenAI format
                    await websocket.send_json({
                        "task": "transcribe",
                        "language": info.language,
                        "duration": total_duration,
                        "segments": all_segments,
                        "text": " ".join(s["text"].strip() for s in all_segments)
                    })
                else:
                    # Send completion signal for streaming format
                    await websocket.send_json({
                        "status": "success",
                        "message": "Stream processing completed",
                        "segments": transcription,
                        "chunk_info": {
                            "is_final": is_final
                        }
                    })

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            await websocket.send_json({
                "status": "error",
                "error": str(e)
            })
        finally:
            # clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    try:
        # initialize audio buffer
        audio_buffer = bytearray()

        while True:
            try:
                # receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=data_timeout_ms / 1000  # convert ms to seconds
                    )
                except asyncio.TimeoutError:
                    # if we have data in buffer and timeout occurred, process
                    # it
                    if len(audio_buffer) > 0:
                        logger.info(
                            f"Data receive timeout ({data_timeout_ms}ms), processing final {
                                len(audio_buffer)} bytes of audio data")
                        await process_audio_chunk(bytes(audio_buffer), is_final=True)
                    break

                # check if it is a disconnect signal
                if message.get("type") == "websocket.disconnect":
                    logger.info("WebSocket connection closed")
                    break

                # process the binary audio data
                if message.get(
                        "type") == "websocket.receive" and "bytes" in message:
                    data = message["bytes"]
                    audio_buffer.extend(data)

                    # check if enough data is collected
                    if is_buffer_ready(audio_buffer, chunk_size):
                        # process the complete audio chunk
                        await process_audio_chunk(bytes(audio_buffer[:chunk_size]))
                        # keep the unprocessed data
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


@app.get("/v1/asr/streaming/status")
async def get_streaming_status():
    """check the status of streaming ASR service"""
    return {
        "ready": faster_asr_ready.is_set() and faster_asr is not None,
        "model_loaded": faster_asr is not None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7066)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--return_timestamps", type=str, default=True)
    parser.add_argument("--compute_type", type=str, default="float32")
    parser.add_argument("--faster_whisper_model", type=str, default="tiny")

    args = parser.parse_args()

    # initiate WhisperModel for non-streaming ASR
    asr = WhisperModel(
        model_name_or_path=args.model_name_or_path,
        language=args.language,
        device=args.device,
        return_timestamps=args.return_timestamps,
    )

    # initiate Faster Whisper model for streaming case in a separate thread
    faster_whisper_thread = threading.Thread(
        target=init_faster_whisper,
        args=(args.faster_whisper_model, args.device, args.compute_type),
        daemon=True
    )
    faster_whisper_thread.start()

    uvicorn.run(app, host=args.host, port=args.port)
