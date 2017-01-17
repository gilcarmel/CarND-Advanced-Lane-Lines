import cv2
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import lane_finder


def generate_output_frame(img):
    global frame_number
    global basename
    left_line, right_line, image_dict, confident = lane_finder.process_image(img)
    final_image = image_dict[lane_finder.FRONT_CAM_WITH_LANE_FILL]
    if frame_number % 10 == 0:
        lane_finder.write_output("{0}/frame_{1:0>4}".format(basename, frame_number), img, image_dict)
        cv2.imwrite('intermediate/{0}/{1:0>4}.jpg'.format(basename, frame_number), final_image)
    frame_number += 1
    return final_image


if __name__ == "__main__":
    basename = "project_video"
    clip = VideoFileClip(basename + ".mp4")
    output_name = basename + "_out.mp4"
    frame_number = 0
    output_clip = clip.fl_image(generate_output_frame)
    output_clip.write_videofile(output_name, audio=False)
