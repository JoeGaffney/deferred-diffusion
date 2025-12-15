from tests.utils import (
    image_to_base64,
    load_json_file,
    save_image_and_assert_file_exists,
    setup_output_file,
)
from workflows.comfy.comfy import main
from workflows.context import WorkflowContext
from workflows.schemas import Patch, WorkflowRequest


def test_text_to_image():
    output_name = setup_output_file("workflows", "text_to_image_v001")
    workflow_path = "../assets/workflows/text_to_image_v001.json"

    patches = [
        Patch(
            title="positive_prompt",
            class_type="PrimitiveStringMultiline",
            value="A beautiful landscape with mountains and a river",
        )
    ]
    result = main(
        WorkflowContext(
            WorkflowRequest(
                workflow=load_json_file(workflow_path),
                patches=patches,
            ),
        )
    )

    save_image_and_assert_file_exists(result, output_name)


def test_image_to_image():
    output_name = setup_output_file("workflows", "image_to_image_v001")
    workflow_path = "../assets/workflows/image_to_image_v001.json"

    patches = [
        Patch(
            title="positive_prompt",
            class_type="PrimitiveStringMultiline",
            value="Change the car color to red, turn the headlights on",
        ),
        Patch(title="Load Image", class_type="LoadImage", value=image_to_base64("../assets/color_v003.png")),
    ]
    result = main(
        WorkflowContext(
            WorkflowRequest(
                workflow=load_json_file(workflow_path),
                patches=patches,
            ),
        )
    )

    save_image_and_assert_file_exists(result, output_name)
