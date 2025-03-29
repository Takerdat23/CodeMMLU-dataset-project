import ast
import os
import math
import base64
import traceback
from io import BytesIO

import cv2
import torch
import imageio
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from transformers import StoppingCriteria


from moviepy.editor import VideoFileClip
import random
import librosa
import soundfile as sf
import torchaudio.compliance.kaldi as ta_kaldi
from subprocess import CalledProcessError, run, Popen, PIPE
import math
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def create_photo_grid(arr, rows=None, cols=None):
    """
    Create a photo grid from a 4D numpy array with shape [t, h, w, c].

    Parameters:
        arr (numpy.ndarray): Input array with shape [t, h, w, c].
        rows (int): Optional. Number of rows in the grid. If not set, it will be determined based on `cols` or the square root of `t`.
        cols (int): Optional. Number of columns in the grid. If not set, it will be determined based on `rows` or the square root of `t`.

    Returns:
        numpy.ndarray: A 3D numpy array representing the photo grid.
    """

    if isinstance(arr, list):
        if isinstance(arr[0], Image.Image):
            arr = np.stack([np.array(img) for img in arr])
        elif isinstance(arr[0], np.ndarray):
            arr = np.stack(arr)
        else:
            raise ValueError("Invalid input type. Expected list of Images or numpy arrays.")

    t, h, w, c = arr.shape
    
    # Calculate the number of rows and columns if not provided
    if rows is None and cols is None:
        rows = math.ceil(math.sqrt(t))
        cols = math.ceil(t / rows)
    elif rows is None:
        rows = math.ceil(t / cols)
    elif cols is None:
        cols = math.ceil(t / rows)

    # Check if the grid can hold all the images
    if rows * cols < t:
        raise ValueError(f"Not enough grid cells ({rows}x{cols}) to hold all images ({t}).")
    
    # Create the grid array with appropriate height and width
    grid_height = h * rows
    grid_width = w * cols
    grid = np.zeros((grid_height, grid_width, c), dtype=arr.dtype)
    
    # Fill the grid with images
    for i in range(t):
        row_idx = i // cols
        col_idx = i % cols
        grid[row_idx*h:(row_idx+1)*h, col_idx*w:(col_idx+1)*w, :] = arr[i]
    
    return grid


def process_image(image_path, processor, aspect_ratio='pad'):
    image = Image.open(image_path).convert('RGB')

    images = [np.array(image)]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return images


def frame_sample(duration, mode='uniform', num_frames=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(duration - 1) / num_frames

        frame_ids = []
        for i in range(num_frames):
            # Calculate the start and end indices of each segment
            start = seg_size * i
            end   = seg_size * (i + 1)
            # Append the middle index of the segment to the list
            frame_ids.append((start + end) / 2)

        return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        # return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def process_audio_file(wav_path):
    # read wav
    #print(wav_path)
    wav, sr = sf.read(wav_path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if len(wav) > 30 * sr:
        max_start = len(wav) - 30 * sr
        start = random.randint(0, max_start)
        wav = wav[start: start + 30 * sr]
    if len(wav) < 30 * sr:
        pad_length = 30 * sr - len(wav)
        wav = np.pad(wav, (0, pad_length), mode='constant', constant_values=0.0)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")

    # beats
    raw_wav = torch.from_numpy(wav).to('cpu')
    waveform = raw_wav.unsqueeze(0) * 2 ** 15
    fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10).to(torch.bfloat16)
    return fbank.unsqueeze(0)

def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
        #print(int(start))
        #print(int(end))
        #print("AAAA")
    return all_clips_timepoints


def tokenizer_token(prompt, tokenizer, return_tensors=None):
    # Tokenize normally
    input_ids = []
    current_chunk = ""

    i = 0
    while i < len(prompt):
            current_chunk += prompt[i]
            i += 1

    if current_chunk:
        input_ids.extend(tokenizer(current_chunk, add_special_tokens=False).input_ids)

    # Convert to PyTorch tensor if requested
    if return_tensors == 'pt':
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
