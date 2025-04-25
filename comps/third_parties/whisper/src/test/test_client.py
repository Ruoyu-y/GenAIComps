import asyncio
import websockets
import pyaudio
import json
import wave
import time
import base64
from typing import Optional

class WhisperStreamingClient:
    def __init__(
        self,
        websocket_url: str = "ws://localhost:7077/v1/asr/streaming",
        chunk_duration_ms: int = 3000,
        frames_per_chunk: int = 16000,
        sample_rate: int = 16000,
        channels: int = 1,
        format: int = pyaudio.paInt16,
        response_type: str = "json"
    ):
        """
        Whisper client
        Args:
            websocket_url: WebSocket url
            chunk_duration_ms: chunk size
            frames_per_chunk
            sample_rate
            channels
            format
        """
        self.websocket_url = websocket_url
        self.chunk_duration_ms = chunk_duration_ms
        self.frames_per_chunk = frames_per_chunk
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.response_type = response_type
        
        # set chunk size
        self.chunk_size = int((chunk_duration_ms * frames_per_chunk) / 1000)
        
        # init PyAudio
        self.audio = pyaudio.PyAudio()
        
    async def process_audio_stream(self, audio_source: Optional[str] = None):
        """
        Args:
            audio_source: path to audio file 
        """
        # setup WebSocket URL
        url = f"{self.websocket_url}"
        
        try:
            async with websockets.connect(url) as websocket:
                print("Connection start")
                
                if audio_source:
                    # get audio from file
                    await self._process_file(websocket, audio_source)
                else:
                    # get audio from microphone
                    await self._process_microphone(websocket)
                    
        except Exception as e:
            print(f"Connection error: {e}")
            
    async def _process_file(self, websocket, audio_file: str):
        """Process audio from file"""
        total_time = 0
        total_chunks = 0
        with wave.open(audio_file, 'rb') as wf:
            while True:
                data = wf.readframes(self.chunk_size)
                if not data:
                    break

                # send audio
                json_blob = {
                    "event_id": "event_001",
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode("utf-8")
                }
                #await websocket.send(data)
                await websocket.send(json.dumps(json_blob))
                start = time.time()

                # receive result
                result = await websocket.recv()
                duration = time.time() - start

                print("duration: %f" % duration)

                #self._handle_result(result)
                result_json = json.loads(result)
                #print(result_json["delta"])
                if result_json["delta"] != "":
                    total_time += duration
                    total_chunks += 1
                else:
                    print("Empty chunk")
                print(result_json)
        print(f"average duration: {total_time/total_chunks}")
                
    async def _process_microphone(self, websocket):
        """process audio from microphone"""
        # start stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("Start recording...")
        
        try:
            while True:
                # get audio data
                data = stream.read(self.chunk_size)
                
                # send audio data
                await websocket.send(data)
                print(time.time())
                
                # get response
                result = await websocket.recv()
                self._handle_result(result)
                
        except KeyboardInterrupt:
            print("\nstop")
        finally:
            stream.stop_stream()
            stream.close()
            
    def _handle_result(self, result: str):
        """Process transcription result"""
        try:
            result_json = json.loads(result)
            if result_json["status"] == "success":
                segments = result_json["segments"]
                for segment in segments:
                    print(f"Result: {segment['text']}")
                print("language: ", result_json["language"])
                print("language: ", result_json["text"])
                #print("chunk info: ", result_json["chunk_info"])
            else:
                print(f"Error: {result_json.get('error', 'unknown error')}")
        except json.JSONDecodeError:
            print(f"Failed to transcribe: {result}")
            
    def close(self):
        """close client"""
        self.audio.terminate()

async def main():
    # init client
    client = WhisperStreamingClient(
        chunk_duration_ms=3000,    # 3s
        sample_rate=16000,
        response_type="verbose_json",
        websocket_url="ws://localhost:7077/v1/realtime?intent=transcription"
    )

    try:
        # get audio from microphone
        #await client.process_audio_stream()

        # get audio from file
        await client.process_audio_stream(file_path_A)
        time.sleep(2)
        await client.process_audio_stream(file_path_B)

    finally:
        client.close()

if __name__ == "__main__":

    file_path_A = "/root/test/output2.wav"
    file_path_B = "/root/test/test01_20s.wav"
    # run client process
    asyncio.run(main())