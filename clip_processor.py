import imageio

imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip
import lane_finder


def generate_output_frame(img):
    left_line, right_line, image_dict, confident = lane_finder.process_image(img)

    return image_dict[lane_finder.FRONT_CAM_WITH_LANE_FILL]


if __name__ == "__main__":
    basename = "project_video"
    clip = VideoFileClip(basename + ".mp4")
    output_name = basename + "_out.mp4"
    output_clip = clip.fl_image(generate_output_frame)
    output_clip.write_videofile(output_name, audio=False)
