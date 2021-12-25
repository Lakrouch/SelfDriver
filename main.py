from pipeline import LaneMemory
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from detecter import object_detection

# For video
input_video = "ru_test_short.mp4"
input_path = "test_videos/"+input_video
white_output = 'test_videos_output/ru_test.mp4'+input_video
black_output = "test_video_black_output/"+input_video
object_detection(input_path, input_video)
clip1 = VideoFileClip(black_output)
laneMem = LaneMemory()
white_clip = clip1.fl_image(laneMem.process) # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))