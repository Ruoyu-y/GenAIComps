# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import uuid
import wave
from fastapi import WebSocket

from faster_whisper import WhisperModel
from whisper_server import BYTES_PER_SAMPLE, SAMPLE_RATE

streaming_asr = None
streaming_asr_ready = threading.Event()


class FasterWhisperModel:
    """Leverage FasterWhisper to do realtime transcription on CPU"""

    def __init__(
        self,
        model_name_or_path="openai/whisper-tiny",
        device="cpu",
        compute_type="int4"
    ) -> None:
        self.model = WhisperModel(
            model_size_or_path=model_name_or_path,
            device=device,
            compute_type=compute_type
        )

    async def audio2text_streaming(self,
                                   websocket: WebSocket,
                                   audio_data: bytes,
                                   item_id: int,
                                   event_id: str,
                                   vad_filter: bool = True,
                                   beam_size: int = 5,
                                   language: str = "en",
                                   min_silence_duration_ms: int = 500,
                                   is_final: bool = False):
        """process the audio data chunk"""
        if not audio_data:
            return
        transcription = ""

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
            segments, _ = self.model.transcribe(
                temp_filename,
                language=None if language == "auto" else language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=min_silence_duration_ms)
            )

            for i in len(segments):
                response = {
                    "content_index": i,
                    "type": "conversation.item.input_audio_transcription.delta",
                    "event_id": event_id,
                    "item_id": "item_" + f"{item_id:03}",
                    "delta": segments[i].text,
                }
                transcription += segments[i].text

                # send the recognition results
                await websocket.send_json(response)

            # if it is the last chunk of data, send the completion signal
            if is_final:
                # Send completion signal for streaming format
                await websocket.send_json({
                    "content_index": 0,
                    "type": "conversation.item.input_audio_transcription.completion",
                    "event_id": event_id,
                    "item_id": "item_" + f"{item_id:03}",
                    "transcript": transcription,
                })
                transcription = ""

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
