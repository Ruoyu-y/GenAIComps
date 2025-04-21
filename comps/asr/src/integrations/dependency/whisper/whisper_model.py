# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
import time
import uuid
import urllib.request
import wave
import webrtcvad
from fastapi import WebSocket, Form

import numpy as np
import torch
from datasets import Audio, Dataset
from pydub import AudioSegment

BYTES_PER_SAMPLE = 2  # 16-bit audio
SAMPLE_RATE = 16000  # Hz


class WhisperModel:
    """Convert audio to text."""

    def __init__(
        self,
        model_name_or_path="openai/whisper-small",
        language="english",
        device="cpu",
        hpu_max_len=8192,
        return_timestamps=False,
    ):
        if device == "hpu":
            # Explicitly link HPU with Torch
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

            adapt_transformers_to_gaudi()
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.asr_model_name_or_path = os.environ.get(
            "ASR_MODEL_PATH", model_name_or_path)
        print("Downloading model: {}".format(self.asr_model_name_or_path))

        if device == "xpu":
            # intel gpu mode
            from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
            from transformers import WhisperProcessor
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name_or_path,
                load_in_4bit=True,
                optimize_model=False,
                use_cache=True
            )
            self.model.to(self.device)
            self.model.forced_decoder_ids = None
            self.processor = WhisperProcessor.from_pretrained(
                model_name_or_path)
            print("Whisper initialized on Intel GPU.")
        else:
            # cpu mode
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name_or_path).to(self.device)
            self.processor = WhisperProcessor.from_pretrained(
                model_name_or_path)
        self.model.eval()

        self.language = language
        self.hpu_max_len = hpu_max_len
        self.return_timestamps = return_timestamps

        if device == "hpu":
            self._warmup_whisper_hpu_graph(
                os.path.dirname(os.path.abspath(__file__)) +
                "/../../../../assets/ljspeech_30s_audio.wav"
            )
            self._warmup_whisper_hpu_graph(
                os.path.dirname(os.path.abspath(__file__)) +
                "/../../../../assets/ljspeech_60s_audio.wav"
            )

    def _audiosegment_to_librosawav(self, audiosegment):
        # https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
        # This way is faster than librosa.load or HuggingFace Dataset wrapper
        # only select the first channel
        channel_sounds = audiosegment.split_to_mono()[:1]
        samples = [s.get_array_of_samples() for s in channel_sounds]

        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        fp_arr = fp_arr.reshape(-1)

        return fp_arr

    def _warmup_whisper_hpu_graph(self, path_to_audio):
        print("[ASR] warmup...")
        waveform = AudioSegment.from_file(path_to_audio).set_frame_rate(16000)
        waveform = self._audiosegment_to_librosawav(waveform)

        try:
            processed_inputs = self.processor(
                waveform,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
                sampling_rate=16000,
            )
        except RuntimeError as e:
            if "Padding size should be less than" in str(e):
                # short-form
                processed_inputs = self.processor(
                    waveform,
                    return_tensors="pt",
                    sampling_rate=16000,
                )
            else:
                raise e

        if processed_inputs.input_features.shape[-1] < 3000:
            # short-form
            processed_inputs = self.processor(
                waveform,
                return_tensors="pt",
                sampling_rate=16000,
            )
        else:
            processed_inputs["input_features"] = torch.nn.functional.pad(
                processed_inputs.input_features,
                (0, self.hpu_max_len - processed_inputs.input_features.size(-1)),
                value=-1.5,
            )
            processed_inputs["attention_mask"] = torch.nn.functional.pad(
                processed_inputs.attention_mask,
                (0, self.hpu_max_len + 1 - processed_inputs.attention_mask.size(-1)),
                value=0,
            )

        _ = self.model.generate(
            **(
                processed_inputs.to(
                    self.device,
                )
            ),
            language=self.language,
            return_timestamps=self.return_timestamps,
        )

    def audio2text(self, audio_path):
        """Convert audio to text.

        audio_path: the path to the input audio, e.g. ~/xxx.mp3
        """
        start = time.time()

        try:
            waveform = AudioSegment.from_file(audio_path).set_frame_rate(16000)
            waveform = self._audiosegment_to_librosawav(waveform)
        except Exception as e:
            print(f"[ASR] audiosegment to librosa wave fail: {e}")
            audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column(
                "audio", Audio(sampling_rate=16000))
            waveform = audio_dataset[0]["audio"]["array"]

        try:
            processed_inputs = self.processor(
                waveform,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
                sampling_rate=16000,
            )
        except RuntimeError as e:
            if "Padding size should be less than" in str(e):
                # short-form
                processed_inputs = self.processor(
                    waveform,
                    return_tensors="pt",
                    sampling_rate=16000,
                )
            else:
                raise e
        if processed_inputs.input_features.shape[-1] < 3000:
            # short-form
            processed_inputs = self.processor(
                waveform,
                return_tensors="pt",
                sampling_rate=16000,
            )
        elif self.device == "hpu" and processed_inputs.input_features.shape[-1] > 3000:
            processed_inputs["input_features"] = torch.nn.functional.pad(
                processed_inputs.input_features,
                (0, self.hpu_max_len - processed_inputs.input_features.size(-1)),
                value=-1.5,
            )
            processed_inputs["attention_mask"] = torch.nn.functional.pad(
                processed_inputs.attention_mask,
                (0, self.hpu_max_len + 1 - processed_inputs.attention_mask.size(-1)),
                value=0,
            )

        predicted_ids = self.model.generate(
            **(
                processed_inputs.to(
                    self.device,
                )
            ),
            language=self.language,
            return_timestamps=self.return_timestamps,
        )
        # pylint: disable=E1101
        result = self.processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True, normalize=True)[0]
        if self.language in ["chinese", "mandarin"]:
            from zhconv import convert

            result = convert(result, "zh-cn")
        print(
            f"generated text in {time.time() - start} seconds, and the result is: {result}")
        return result

    async def audio2text_streaming(self,
                                   websocket: WebSocket,
                                   audio_data: bytes,
                                   event_id: str,
                                   item_id: int,
                                   language: str = "en",
                                   is_final: bool = False):
        """Convert streaming audio to text"""
        if not audio_data:
            return

        transcription = ""
        vad = webrtcvad.Vad()
        vad.set_mode(2) # set vad agressiveness mode

        def detect_audio_segments(audio, sample_rate):
            """detect speech segments using VAD"""
            filtered_segments = []
            default_chunk = int(sample_rate * 30 / 1000 * 2) # check VAD with 30ms chunk
            start = 0
            while (start + default_chunk <= len(audio)):
                frame = audio[start:start+default_chunk]
                try:
                    if vad.is_speech(frame, sample_rate):
                        filtered_segments.append(frame)
                except Exception as e:
                    print(f"[ASR] Failed to perform VAD check with {e}")
                    break
                start += default_chunk
            
            if start < len(audio):
                padding = b'\x00' * (default_chunk - len(audio) + start + 1)
                frame = audio[start:len(audio)-1] + padding
                try:
                    if vad.is_speech(frame, sample_rate):
                        filtered_segments.append(frame)
                except Exception as e:
                    print(f"[ASR] Failed to perform VAD check the remaining frame with {e}")

            # concatenate all speech segments
            if filtered_segments:
                return b''.join(filtered_segments)
            else:
                return None
        
        # skip silence using VAD
        filtered_audio_data = detect_audio_segments(audio_data, SAMPLE_RATE)
        if filtered_audio_data is None:
            print(f"[ASR] Skip empty audio data.")
            await websocket.send_json({
                "type": "conversation.item.input_audio_transcription.delta",
                    "event_id": event_id,
                    "item_id": "item_" + f"{item_id:03}",
                    "content_index": 0,
                    "delta": ""
            })
        else :
            # generate a temporary file name
            uid = str(uuid.uuid4())
            temp_filename = f"{uid}.wav"

            try:
                # create a WAV file
                with wave.open(temp_filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # single channel
                    wav_file.setsampwidth(BYTES_PER_SAMPLE)  # 16-bit
                    wav_file.setframerate(SAMPLE_RATE)
                    wav_file.writeframes(filtered_audio_data)
            except Exception as e:
                print(f"[ASR] Failed to save audiosegment to file: {e}")
                return

            ds = Dataset.from_dict({"audio": [temp_filename]}).cast_column("audio", Audio())

            if self.device == "xpu" and filtered_audio_data is not None:
                with torch.inference_mode():
                    sample = ds[0]["audio"]

                    inputs = self.processor(sample["array"],
                                       sampling_rate=sample["sampling_rate"],
                                       return_attention_mask=True,
                                       return_tensors="pt")
                    input_features = inputs.input_features.to('xpu')
                    attention_mask = inputs.attention_mask.to('xpu')
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                    predicted_ids = self.model.generate(input_features,
                                                   forced_decoder_ids=forced_decoder_ids,
                                                   attention_mask=attention_mask)
                    output_str = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

                    # change the output format to be compatible with the OpenAI API
                    '''
                    if is_final:
                        response = {
                            "event_id": event_id,
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "item_" + f"{item_id:03}",
                            "content_index": 0,
                            "transcript": transcription + output_str
                        }
                        transcription = ""
                    else:
                    '''
                    response = {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "event_id": event_id,
                        "item_id": "item_" + f"{item_id:03}",
                        "content_index": 0,
                        "delta": output_str
                    }
                    # combine delta together
                    # transcription += output_str

                    # if the output is in Chinese, convert it to simplified Chinese
                    if self.language in ["chinese", "mandarin"]:
                        from zhconv import convert
                        if "delta" in response:
                            response["delta"] = convert(response["delta"], "zh-cn")
                        '''
                        else:
                            response["transcript"] = convert(
                                response["transcript"], "zh-cn")
                        '''

                    # send the response through websocket
                    await websocket.send_json(response)

            else:
                raise ValueError(f"Unsupported device: {self.device}")
            os.remove(temp_filename)


if __name__ == "__main__":
    asr = WhisperModel(
        model_name_or_path="openai/whisper-small", language="english", device="cpu", return_timestamps=True
    )

    # Test multilanguage asr
    asr.language = "chinese"
    urllib.request.urlretrieve(
        "https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/labixiaoxin.wav",
        "sample.wav",
    )
    text = asr.audio2text("sample.wav")

    asr.language = "english"
    urllib.request.urlretrieve(
        "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav",
        "sample.wav",
    )
    text = asr.audio2text("sample.wav")
