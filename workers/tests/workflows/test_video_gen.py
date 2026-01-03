from tests.utils import asset_outputs_exists, image_to_base64, load_json_file
from workflows.comfy.comfy_workflow import main
from workflows.context import WorkflowContext
from workflows.schemas import Patch, WorkflowRequest


def test_video_wan2_2_14B_fun_control():
    workflow_path = "../assets/workflows/video_wan2_2_14B_fun_control_v001.json"

    patches = [
        Patch(
            title="positive_prompt",
            class_type="PrimitiveStringMultiline",
            value="A man in a tuxedo is waving at the camera.",
        ),
        Patch(title="Load Image", class_type="LoadImage", value=image_to_base64("../assets/act_char_v001.png")),
        Patch(title="Load Video", class_type="LoadVideo", value=image_to_base64("../assets/act_reference_v001.mp4")),
    ]
    context = WorkflowContext(
        WorkflowRequest(
            workflow=load_json_file(workflow_path),
            patches=patches,
        ),
        task_id="video_wan2_2_14B_fun_control",
    )
    result = main(context)
    asset_outputs_exists(result)
