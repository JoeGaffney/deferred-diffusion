import logging

logger = logging.getLogger(__name__)


def execute_wan_2_1(params_dict: dict, context) -> dict:
    """
    Video generation worker logic.
    """
    logger.info(f"Executing Wan 2.1 Video with params: {params_dict}")

    generated_path = "/tmp/wan_mock_video.mp4"

    return {
        "output_files": [generated_path],
        "output_text": [],
        "logs": ["Init Wan2.1", f"Rendered {params_dict.get('duration_seconds')}s video"],
    }
