#!/usr/bin/env python3
import pickle
import time
import depthai as dai
import os
import urllib.request


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
        download_url = GoogleDownloader._get_download_url(url)
        urllib.request.urlretrieve(download_url, filename, reporthook=GoogleDownloader.show_progress)
        print("\nDownload completed.")
        return filename

    @staticmethod
    def _get_download_url(url):
        if 'drive.google.com' in url:
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            else:
                raise ValueError("Invalid Google Drive URL format")
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        else:
            download_url = url
        return download_url
    
    @staticmethod
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        else:
            print(f"\rDownloading: {downloaded} bytes", end='', flush=True)


file_path = "encoded_stream.pkl"
if not os.path.exists(file_path):
    print("Prerecorded stream not found, downloading...")
    GoogleDownloader.download(
        "https://drive.google.com/file/d/1vuYwyJazQmhvkMMyrdmVuoXFU-gKQbRg/view?usp=drive_link",
        file_path
    )
visualizer = dai.RemoteConnection()
stream_queue = visualizer.addTopic("H.264 Stream")
reader = PickleReader(file_path)
frame_sender = FrameSender(
    fps=10, # TODO: This is where we want to have stable 30 FPS
    queue=stream_queue
)
try:
    for frame in reader:
        encoded_frame = EncodedFrameMapper.get(frame)
        frame_sender.send(encoded_frame)
except KeyboardInterrupt:
    print("Interrupted by user, stopping...")
finally:
    visualizer.removeTopic("H.264 Stream")
    reader.close()
