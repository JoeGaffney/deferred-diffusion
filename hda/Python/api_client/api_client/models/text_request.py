from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="TextRequest")


@_attrs_define
class TextRequest:
    """
    Attributes:
        negative_prompt (Union[Unset, str]):  Default: 'worst quality, inconsistent motion, blurry, jittery, distorted'.
        num_frames (Union[Unset, int]):  Default: 48.
        prompt (Union[Unset, str]):  Default: 'Detailed, 8k, photorealistic'.
        seed (Union[Unset, int]):  Default: 42.
        model (Union[Unset, str]):  Default: 'qwen_2_5_vl_instruct'.
        messages (Union[Unset, list[Any]]):
    """

    negative_prompt: Union[Unset, str] = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_frames: Union[Unset, int] = 48
    prompt: Union[Unset, str] = "Detailed, 8k, photorealistic"
    seed: Union[Unset, int] = 42
    model: Union[Unset, str] = "qwen_2_5_vl_instruct"
    messages: Union[Unset, list[Any]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        negative_prompt = self.negative_prompt

        num_frames = self.num_frames

        prompt = self.prompt

        seed = self.seed

        model = self.model

        messages: Union[Unset, list[Any]] = UNSET
        if not isinstance(self.messages, Unset):
            messages = self.messages

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if negative_prompt is not UNSET:
            field_dict["negative_prompt"] = negative_prompt
        if num_frames is not UNSET:
            field_dict["num_frames"] = num_frames
        if prompt is not UNSET:
            field_dict["prompt"] = prompt
        if seed is not UNSET:
            field_dict["seed"] = seed
        if model is not UNSET:
            field_dict["model"] = model
        if messages is not UNSET:
            field_dict["messages"] = messages

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        negative_prompt = d.pop("negative_prompt", UNSET)

        num_frames = d.pop("num_frames", UNSET)

        prompt = d.pop("prompt", UNSET)

        seed = d.pop("seed", UNSET)

        model = d.pop("model", UNSET)

        messages = cast(list[Any], d.pop("messages", UNSET))

        text_request = cls(
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            prompt=prompt,
            seed=seed,
            model=model,
            messages=messages,
        )

        text_request.additional_properties = d
        return text_request

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
