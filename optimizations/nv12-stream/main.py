#!/usr/bin/env python3
import pickle
import time
import depthai as dai
import os
import gdown


class PickleReader:
    def __init__(self, path: str) -> None:
        self._file = open(path, "rb")

    def close(self):
        self._file.close()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            data = self.read()
        except EOFError:
            raise StopIteration
        return data

    def read(self):
        return pickle.load(self._file)
        

class ImgFrameMapper:
    @staticmethod
    def get(frame: dict) -> dai.ImgFrame:
        img_frame = dai.ImgFrame()
        img_frame.setData(frame['data'])
        img_frame.setTimestamp(frame['timestamp'])
        img_frame.setSequenceNum(frame['sequence_num'])
        img_frame.setWidth(frame['width'])
        img_frame.setHeight(frame['height'])
        img_frame.setType(frame['type'])
        img_frame.setCategory(frame['category'])
        img_frame.setInstanceNum(frame['instance_num'])
        img_frame.setStride(frame['stride'])
        img_frame.setTimestampDevice(frame['timestamp_device'])
        return img_frame


class FrameSender:
    def __init__(self, fps: float, queue: dai.MessageQueue):
        self._interval = 1.0 / fps
        self._last_sent = 0.0
        self._queue = queue

    def send(self, frame: dai.Buffer):
        current_time = time.time()
        if current_time - self._last_sent < self._interval:
            time.sleep(self._interval - (current_time - self._last_sent))
        self._queue.send(frame)
        self._last_sent = time.time()


class GoogleDownloader:
    @staticmethod
    def download(url: str, filename: str):
        try:
            file_id = GoogleDownloader._get_file_id(url)
            GoogleDownloader._download(filename, file_id)
            return filename
        except Exception as e:
            print(f"Download failed: {e}")
            raise
    
    @staticmethod
    def _download(filename, file_id):
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from Google Drive (File ID: {file_id})...")
        gdown.download(download_url, filename, quiet=False)
        print(f"Download completed: {filename}")

    @staticmethod
    def _get_file_id(url):
        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        else:
            raise ValueError("Invalid Google Drive URL format")
        return file_id


file_path = "nv12_stream.pkl"
if not os.path.exists(file_path):
    print("Prerecorded stream not found, downloading...")
    GoogleDownloader.download(
        "https://drive.google.com/file/d/1qEb2jVZgfLs5dWbD12p58qMq4oGcdWqy/view?usp=drive_link",
        file_path
    )
visualizer = dai.RemoteConnection()
stream_queue = visualizer.addTopic("NV12 Stream")
reader = PickleReader(file_path)
frame_sender = FrameSender(
    fps=30,
    queue=stream_queue
)
counter = 0
try:
    for frame in reader:
        img_frame = ImgFrameMapper.get(frame)
        frame_sender.send(img_frame)
        counter += 1
        if counter % 30 == 0:
            print(f"Sent 30 frames. Time: {time.time()}")
except KeyboardInterrupt:
    print("Interrupted by user, stopping...")
finally:
    visualizer.removeTopic("NV12 Stream")
    reader.close()
