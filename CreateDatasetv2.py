from pydub import AudioSegment
import os
import torch
import uuid
import numpy as np
from diffusers import AudioDiffusionPipeline, DDIMScheduler, UNet2DModel, Mel
import string


class AudioProcessor:
    def __init__(self, input_dir, output_dir="data", chunk_length_ms=6000):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_length_ms = chunk_length_ms
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.unet = UNet2DModel.from_pretrained("teticio/audio-diffusion-256", subfolder="unet")
        self.scheduler = DDIMScheduler.from_pretrained("teticio/audio-diffusion-256", subfolder="scheduler")
        self.mel = Mel()

        self.pipe = AudioDiffusionPipeline(
            vqvae=None,  # Assuming no VQ-VAE is needed; adjust if required
            unet=self.unet,
            mel=self.mel,
            scheduler=self.scheduler,
        )

        # Ensure output directories exist
        self.audio_chunk_dir = os.path.join(self.output_dir, "audio_chunks")
        self.mel_output_dir = os.path.join(self.output_dir, "mel_spectrograms")
        os.makedirs(self.audio_chunk_dir, exist_ok=True)
        os.makedirs(self.mel_output_dir, exist_ok=True)
    
    def is_audio_file(self, file_path):
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a')
        return file_path.lower().endswith(audio_extensions)

    def generate_unique_id(self):
        # Generate a 6-character unique ID
        return uuid.uuid4().hex[:6]

    def split_audio(self, file_path, track_prefix):
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Calculate the number of chunks
        num_chunks = len(audio) // self.chunk_length_ms
        if num_chunks == 0:  # Ensure at least one chunk
            num_chunks = 1

        # Split the audio into chunks and save them with a track-specific prefix
        for i in range(num_chunks):
            start_time = i * self.chunk_length_ms
            end_time = start_time + self.chunk_length_ms
            chunk = audio[start_time:end_time]
            
            # Add fade in and fade out
            chunk = chunk.fade_in(5).fade_out(5)
        
            # Export the chunk with a unique name using the track prefix
            chunk_name = f"{track_prefix}_chunk_{i + 1}.wav"
            chunk.export(os.path.join(self.audio_chunk_dir, chunk_name), format="wav")
        
        return self.audio_chunk_dir
    
    def audio_to_mel(self, chunk_dir):
        # Process each audio chunk and convert it to mel spectrogram
        for root, _, files in os.walk(chunk_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if self.is_audio_file(file_path):
                    self.mel.load_audio(file_path)
                    im = self.mel.audio_slice_to_image(0)
                    
                    # Use the same chunk name for the spectrogram
                    base_name = os.path.splitext(file)[0]
                    unique_filename = f"{base_name}.png"
                    im.save(os.path.join(self.mel_output_dir, unique_filename))
                    print(f"Saved mel spectrogram: {unique_filename}")

    def process_directory(self):
        # Process all files in the input directory
        for root, _, files in os.walk(self.input_dir):
            # Avoid processing subdirectories recursively
            if root != self.input_dir:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                if self.is_audio_file(file_path):
                    # Generate a unique 6-character ID for each track
                    track_prefix = self.generate_unique_id()
                    
                    # Split the audio file into chunks
                    chunk_dir = self.split_audio(file_path, track_prefix)
                    # Convert the audio chunks to mel-spectrograms
                    self.audio_to_mel(chunk_dir)


