import os
import sys
from datasets import load_dataset

sys.path.append(os.getcwd())

from vividbot.data.processor.upload_hf import Uploader

uploader = Uploader()
path = "/home/duytran/Downloads/output_ds/videoinstruck100_vi.json"

# uploader.upload_dir(dir_path=path, 
#                     repo_id="Vividbot/vast2m_vi", 
#                     path_in_repo="metadata", 
#                     repo_type="dataset", overwrite=True)

uploader.upload_file(file_path=path,
                     repo_id= "Vividbot/videoinstruck100k",
                     path_in_repo= "videoinstruck100_vi.json",
                     repo_type= "dataset",
                     overwrite= True)

# for folder in os.listdir(path):
#     uploader.zip_and_upload_dir(
#         dir_path=f"{path}/{folder}",
#         repo_id="Vividbot/instruct500k_vi",
#         path_in_repo=f"images/{folder}.zip",
#         repo_type="dataset",
#         overwrite=False,
#     )



# from huggingface_hub import HfFileSystem
# import numpy as np
# import av
# import io
# import os
# import numpy as np
# import decord
# from matplotlib import pyplot as plt
# from datasets import load_dataset
# fs = HfFileSystem()
# # print all file in .zip file
# temp = fs.open("datasets/Vividbot/vast2m_vi/video/shard_0.zip")
# import zipfile
# # def extract_frames(video_bytes, num_frames=8):
# #     # Create a memory-mapped file from the bytes
# #     container = av.open(io.BytesIO(video_bytes))
# #     # Find the video stream
# #     visual_stream = next(iter(container.streams.video), None)
# #     if not visual_stream:
# #         return None, None
# #     # Extract video properties
# #     video_fps = visual_stream.average_rate
# #     # Initialize arrays to store frames
# #     frames_array = []
# #     # Extract frames
# #     for packet in container.demux([visual_stream]):
# #         for frame in packet.decode():
# #             img_array = np.array(frame.to_image())
# #             frames_array.append(img_array)
# #     return np.array(frames_array)

# def extract_frames(video_bytes, num_frames=8):
#     # Create a memory-mapped file from the bytes
#     container = av.open(io.BytesIO(video_bytes))
    
#     # Find the video stream
#     visual_stream = next(iter(container.streams.video), None)
#     if not visual_stream:
#         return None

#     # Extract video properties
#     total_frames = visual_stream.frames
#     print(f"Total frames in video: {total_frames}")

#     # Calculate the interval to capture the frames
#     interval = max(total_frames // num_frames, 1)
#     print(f"Frame capture interval: {interval}")

#     # Initialize arrays to store frames
#     frames_array = []
#     frame_indices = set(range(0, total_frames, interval))  # Indices of frames to capture
#     frame_counter = 0

#     # Extract frames
#     for packet in container.demux([visual_stream]):
#         for frame in packet.decode():
#             if frame_counter in frame_indices:
#                 img_array = np.array(frame.to_image())
#                 frames_array.append(img_array)
#                 if len(frames_array) >= num_frames:
#                     break
#             frame_counter += 1
#         if len(frames_array) >= num_frames:
#             break
    
#     return np.array(frames_array)
# import cv2
# with zipfile.ZipFile(temp, 'r') as zip_ref:
#     #zip_ref.printdir()
#     # extract all files
#     name1 = zip_ref.namelist()[1]
#     print(name1)
#     video = zip_ref.read(name1)
#     #video = io.BytesIO(video)
#     frames = extract_frames(video)
#     for frame in frames:
#         cv2.imshow('frame', frame)
#         cv2.waitKey(0)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
    

        