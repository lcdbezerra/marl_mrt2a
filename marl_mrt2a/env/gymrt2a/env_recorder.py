import pygame
import numpy as np
import math
import imageio

class VideoRecorder:
    def __init__(self, record_path, record_fps=3):
        if record_path is True:
            record_path = 'recording.gif'
        self.record_path = record_path
        self.record_fps = record_fps
        self.record_ms = 1000*1/self.record_fps
        self.record_frames = []

    def reset(self):
        self.save()
        self.record_frames = []

    def tick(self, frame):
        frame = np.rot90(frame, 3)
        frame = np.fliplr(frame)
        self.record_frames.append(frame.astype('uint8'))

    def save(self):
        try:
            imageio.mimsave(self.record_path, self.record_frames, duration=self.record_ms)
            self.record_frames = []
            print('Recording saved to', self.record_path)
            return True
        except Exception as e:
            print('Error saving recording:', e)
            return False