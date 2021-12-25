import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation


def object_detection(path, name):
    segment_video = instance_segmentation()
    segment_video.load_model("mask_rcnn_coco.h5")

    target_class = segment_video.select_target_classes(person=True, car=True, bus=True)

    segment_video.process_video(
        video_path=path,
        segment_target_classes=target_class,
        output_video_name="test_video_black_output/" + name,
        frames_per_second=15
    )
