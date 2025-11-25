from runwayml import RunwayML

from common.logger import logger
from videos.context import VideoContext


def main(context: VideoContext):
    if context.data.video is None:
        raise ValueError("Input video is None. Please provide a valid video.")

    client = RunwayML()

    video_uri = f"data:video/mp4;base64,{context.get_compressed_video()}"

    try:
        task = client.video_upscale.create(
            model="upscale_v1",
            video_uri=video_uri,
        ).wait_for_task_output()
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")
