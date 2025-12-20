from tests.utils import image_to_base64, load_json_file
from tests.workflows.helpers import save_and_assert_workflow_outputs
from workflows.comfy.comfy_workflow import main
from workflows.context import WorkflowContext
from workflows.schemas import Patch, WorkflowRequest


def test_text_to_image():
    workflow_path = "../assets/workflows/text_to_image_v001.json"

    patches = [
        Patch(
            title="positive_prompt",
            class_type="PrimitiveStringMultiline",
            value="A beautiful landscape with mountains and a river",
        )
    ]
    context = WorkflowContext(
        WorkflowRequest(
            workflow=load_json_file(workflow_path),
            patches=patches,
        ),
    )
    result = main(context)

    save_and_assert_workflow_outputs(context, result, "text_to_image_v001")


def test_image_to_image():
    workflow_path = "../assets/workflows/image_to_image_v001.json"

    patches = [
        Patch(
            title="positive_prompt",
            class_type="PrimitiveStringMultiline",
            value="Change the car color to red, turn the headlights on",
        ),
        Patch(title="Load Image", class_type="LoadImage", value=image_to_base64("../assets/color_v003.png")),
    ]
    context = WorkflowContext(
        WorkflowRequest(
            workflow=load_json_file(workflow_path),
            patches=patches,
        ),
    )
    result = main(context)

    save_and_assert_workflow_outputs(context, result, "image_to_image_v001")


def test_flux_kontext_dev_basic():
    workflow_path = "../assets/workflows/flux_kontext_dev_basic.json"

    patches = [
        Patch(
            title="positive_prompt",
            class_type="PrimitiveStringMultiline",
            value="Change the car color to red, turn the headlights on",
        ),
        Patch(title="Load Image", class_type="LoadImage", value=image_to_base64("../assets/color_v003.png")),
    ]
    context = WorkflowContext(
        WorkflowRequest(
            workflow=load_json_file(workflow_path),
            patches=patches,
        ),
    )
    result = main(context)

    save_and_assert_workflow_outputs(context, result, "flux_kontext_dev_basic")
