from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ImageRequest")


@_attrs_define
class ImageRequest:
    """
    Attributes:
        model (str):
        controlnets (Union[Unset, list[Any]]):
        disable_text_encoder_3 (Union[Unset, bool]):  Default: True.
        guidance_scale (Union[Unset, float]):  Default: 10.0.
        inpainting_full_image (Union[Unset, bool]):  Default: True.
        input_image_path (Union[Unset, str]):  Default: ''.
        input_mask_path (Union[Unset, str]):  Default: ''.
        max_height (Union[Unset, int]):  Default: 2048.
        max_width (Union[Unset, int]):  Default: 2048.
        negative_prompt (Union[Unset, str]):  Default: 'worst quality, inconsistent motion, blurry, jittery, distorted'.
        num_frames (Union[Unset, int]):  Default: 48.
        num_inference_steps (Union[Unset, int]):  Default: 25.
        output_image_path (Union[Unset, str]):  Default: ''.
        prompt (Union[Unset, str]):  Default: 'Detailed, 8k, photorealistic'.
        seed (Union[Unset, int]):  Default: 42.
        strength (Union[Unset, float]):  Default: 0.5.
    """

    model: str
    controlnets: Union[Unset, list[Any]] = UNSET
    disable_text_encoder_3: Union[Unset, bool] = True
    guidance_scale: Union[Unset, float] = 10.0
    inpainting_full_image: Union[Unset, bool] = True
    input_image_path: Union[Unset, str] = ""
    input_mask_path: Union[Unset, str] = ""
    max_height: Union[Unset, int] = 2048
    max_width: Union[Unset, int] = 2048
    negative_prompt: Union[Unset, str] = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_frames: Union[Unset, int] = 48
    num_inference_steps: Union[Unset, int] = 25
    output_image_path: Union[Unset, str] = ""
    prompt: Union[Unset, str] = "Detailed, 8k, photorealistic"
    seed: Union[Unset, int] = 42
    strength: Union[Unset, float] = 0.5
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        model = self.model

        controlnets: Union[Unset, list[Any]] = UNSET
        if not isinstance(self.controlnets, Unset):
            controlnets = self.controlnets

        disable_text_encoder_3 = self.disable_text_encoder_3

        guidance_scale = self.guidance_scale

        inpainting_full_image = self.inpainting_full_image

        input_image_path = self.input_image_path

        input_mask_path = self.input_mask_path

        max_height = self.max_height

        max_width = self.max_width

        negative_prompt = self.negative_prompt

        num_frames = self.num_frames

        num_inference_steps = self.num_inference_steps

        output_image_path = self.output_image_path

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
        if controlnets is not UNSET:
            field_dict["controlnets"] = controlnets
        if disable_text_encoder_3 is not UNSET:
            field_dict["disable_text_encoder_3"] = disable_text_encoder_3
        if guidance_scale is not UNSET:
            field_dict["guidance_scale"] = guidance_scale
        if inpainting_full_image is not UNSET:
            field_dict["inpainting_full_image"] = inpainting_full_image
        if input_image_path is not UNSET:
            field_dict["input_image_path"] = input_image_path
        if input_mask_path is not UNSET:
            field_dict["input_mask_path"] = input_mask_path
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
        if output_image_path is not UNSET:
            field_dict["output_image_path"] = output_image_path
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

        controlnets = cast(list[Any], d.pop("controlnets", UNSET))

        disable_text_encoder_3 = d.pop("disable_text_encoder_3", UNSET)

        guidance_scale = d.pop("guidance_scale", UNSET)

        inpainting_full_image = d.pop("inpainting_full_image", UNSET)

        input_image_path = d.pop("input_image_path", UNSET)

        input_mask_path = d.pop("input_mask_path", UNSET)

        max_height = d.pop("max_height", UNSET)

        max_width = d.pop("max_width", UNSET)

        negative_prompt = d.pop("negative_prompt", UNSET)

        num_frames = d.pop("num_frames", UNSET)

        num_inference_steps = d.pop("num_inference_steps", UNSET)

        output_image_path = d.pop("output_image_path", UNSET)

        prompt = d.pop("prompt", UNSET)

        seed = d.pop("seed", UNSET)

        strength = d.pop("strength", UNSET)

        image_request = cls(
            model=model,
            controlnets=controlnets,
            disable_text_encoder_3=disable_text_encoder_3,
            guidance_scale=guidance_scale,
            inpainting_full_image=inpainting_full_image,
            input_image_path=input_image_path,
            input_mask_path=input_mask_path,
            max_height=max_height,
            max_width=max_width,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            output_image_path=output_image_path,
            prompt=prompt,
            seed=seed,
            strength=strength,
        )

        image_request.additional_properties = d
        return image_request

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
