import imageio

imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip
import lane_finder
import cv2
# import numpy as np

def generate_output_frame(img):
    result = lane_finder.process_image(img)
    return result[lane_finder.FRONT_CAM_WITH_LANE_FILL]
    # return np.zeros((128,128,3), np.uint8)


if __name__ == "__main__":
    basename = "project_video"
    clip = VideoFileClip(basename + ".mp4")
    output_name = basename + "_out.mp4"
    output_clip = clip.fl_image(generate_output_frame)
    output_clip.write_videofile(output_name, audio=False)
