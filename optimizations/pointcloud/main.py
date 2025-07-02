#!/usr/bin/env python3
import pickle
import time
from typing import Protocol
import depthai as dai
import os
import gdown
import threading


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
        

class Mapper(Protocol):
    @staticmethod
    def get(frame: dict) -> dai.Buffer:
        pass


class EncodedFrameMapper:
    @staticmethod
    def get(frame: dict) -> dai.EncodedFrame:
        encoded_frame = dai.EncodedFrame()
        encoded_frame.setData(frame['data'])
        encoded_frame.setTimestamp(frame['timestamp'])
        encoded_frame.setSequenceNum(frame['sequence_num'])
        encoded_frame.setProfile(frame['profile'])
        encoded_frame.setQuality(frame['quality'])
        encoded_frame.setLossless(frame['lossless'])
        encoded_frame.setFrameType(frame['frame_type'])
        encoded_frame.setBitrate(frame['bitrate'])
        encoded_frame.setWidth(frame['width'])
        encoded_frame.setHeight(frame['height'])
        encoded_frame.setTimestampDevice(frame['timestamp_device'])
        return encoded_frame


class ImgFrameMapper:
    @staticmethod
    def get(frame: dict) -> dai.ImgFrame:
        img_frame = dai.ImgFrame()
        img_frame.setData(frame['data'])
        img_frame.setTimestamp(frame['timestamp'])
        img_frame.setTimestampDevice(frame['timestamp_device'])
        img_frame.setSequenceNum(frame['sequence_num'])
        img_frame.setWidth(frame['width'])
        img_frame.setHeight(frame['height'])
        img_frame.setType(frame['type'])
        img_frame.setStride(frame['stride'])
        img_frame.setCategory(frame['category'])
        img_frame.setInstanceNum(frame['instance_num'])
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


def ensure_downloaded(url: str, file_name: str):
    if os.path.exists(file_name):
        return
    print(f"Prerecorded stream ({file_name}) not found, downloading...")
    GoogleDownloader.download(
        url,
        file_name
    )

def send_frames(file_path: str, queue: dai.MessageQueue, mapper: Mapper):
    reader = PickleReader(file_path)
    frame_sender = FrameSender(
        fps=30, 
        queue=queue
    )
    try:
        for frame in reader:
            message = mapper.get(frame)
            frame_sender.send(message)
    except dai.MessageQueue.QueueException:
        pass
    finally:
        reader.close()

encoded_stream_path = "encoded_stream.pkl"
ensure_downloaded(
    url="https://drive.google.com/file/d/1AC5SdIRF4d5pZIiu9wRdi9Wjak6zVp9O/view?usp=drive_link",
    file_name=encoded_stream_path,
)
depth_stream_path = "depth_stream.pkl"
ensure_downloaded(
    url="https://drive.google.com/file/d/1DmBXj9pqR8NFZo12QAeG9KeynKtPMZDD/view?usp=drive_link",
    file_name=depth_stream_path,
)

visualizer = dai.RemoteConnection()
encoded_stream_queue = visualizer.addTopic("_Point Cloud Color")
depth_queue = visualizer.addTopic("Point Cloud")

encoded_stream_thread = threading.Thread(
    target=lambda: send_frames(
        file_path=encoded_stream_path,
        queue=encoded_stream_queue,
        mapper=EncodedFrameMapper
    ),
    daemon=False
)
depth_stream_thread = threading.Thread(
    target=lambda: send_frames(
        file_path=depth_stream_path,
        queue=depth_queue,
        mapper=ImgFrameMapper
    ),
    daemon=False
)
encoded_stream_thread.start()
depth_stream_thread.start()
start = time.time()
try:
    depth_stream_thread.join()
    encoded_stream_thread.join()
except KeyboardInterrupt:
    print("Process got interrupted, stopping threads...")
finally:
    visualizer.removeTopic("_Point Cloud Color")
    visualizer.removeTopic("Point Cloud")
print(f"Finished sending frames. Time: {time.time() - start:.2f} seconds")