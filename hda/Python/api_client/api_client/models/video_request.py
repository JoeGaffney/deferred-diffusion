from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="VideoRequest")


@_attrs_define
class VideoRequest:
    """
    Attributes:
        model (str):
        input_image_path (Union[Unset, str]):  Default: '../tmp/input.png'.
        max_height (Union[Unset, int]):  Default: 2048.
        max_width (Union[Unset, int]):  Default: 2048.
        negative_prompt (Union[Unset, str]):  Default: 'worst quality, inconsistent motion, blurry, jittery, distorted'.
        num_frames (Union[Unset, int]):  Default: 48.
        num_inference_steps (Union[Unset, int]):  Default: 25.
        output_video_path (Union[Unset, str]):  Default: '../tmp/outputs/processed.mp4'.
        prompt (Union[Unset, str]):  Default: 'Detailed, 8k, photorealistic'.
        seed (Union[Unset, int]):  Default: 42.
        strength (Union[Unset, float]):  Default: 0.5.
    """

    model: str
    input_image_path: Union[Unset, str] = "../tmp/input.png"
    max_height: Union[Unset, int] = 2048
    max_width: Union[Unset, int] = 2048
    negative_prompt: Union[Unset, str] = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_frames: Union[Unset, int] = 48
    num_inference_steps: Union[Unset, int] = 25
    output_video_path: Union[Unset, str] = "../tmp/outputs/processed.mp4"
    prompt: Union[Unset, str] = "Detailed, 8k, photorealistic"
    seed: Union[Unset, int] = 42
    strength: Union[Unset, float] = 0.5
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        model = self.model

        input_image_path = self.input_image_path

        max_height = self.max_height

        max_width = self.max_width

        negative_prompt = self.negative_prompt

        num_frames = self.num_frames

        num_inference_steps = self.num_inference_steps

        output_video_path = self.output_video_path

        prompt = self.prompt

        seed = self.seed

        strength = self.strength

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "model": model,
            }
        )
        if input_image_path is not UNSET:
            field_dict["input_image_path"] = input_image_path
        if max_height is not UNSET:
            field_dict["max_height"] = max_height
        if max_width is not UNSET:
            field_dict["max_width"] = max_width
        if negative_prompt is not UNSET:
            field_dict["negative_prompt"] = negative_prompt
        if num_frames is not UNSET:
            field_dict["num_frames"] = num_frames
        if num_inference_steps is not UNSET:
            field_dict["num_inference_steps"] = num_inference_steps
        if output_video_path is not UNSET:
            field_dict["output_video_path"] = output_video_path
        if prompt is not UNSET:
            field_dict["prompt"] = prompt
        if seed is not UNSET:
            field_dict["seed"] = seed
        if strength is not UNSET:
            field_dict["strength"] = strength

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        model = d.pop("model")

        input_image_path = d.pop("input_image_path", UNSET)

        max_height = d.pop("max_height", UNSET)

        max_width = d.pop("max_width", UNSET)

        negative_prompt = d.pop("negative_prompt", UNSET)

        num_frames = d.pop("num_frames", UNSET)

        num_inference_steps = d.pop("num_inference_steps", UNSET)

        output_video_path = d.pop("output_video_path", UNSET)

        prompt = d.pop("prompt", UNSET)

        seed = d.pop("seed", UNSET)

        strength = d.pop("strength", UNSET)

        video_request = cls(
            model=model,
            input_image_path=input_image_path,
            max_height=max_height,
            max_width=max_width,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            output_video_path=output_video_path,
            prompt=prompt,
            seed=seed,
            strength=strength,
        )

        video_request.additional_properties = d
        return video_request

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
