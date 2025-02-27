from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TextResponse")


@_attrs_define
class TextResponse:
    """
    Attributes:
        response (str):
        chain_of_thought (list[Any]):
    """

    response: str
    chain_of_thought: list[Any]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        response = self.response

        chain_of_thought = self.chain_of_thought

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "response": response,
                "chain_of_thought": chain_of_thought,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        response = d.pop("response")

        chain_of_thought = cast(list[Any], d.pop("chain_of_thought"))

        text_response = cls(
            response=response,
            chain_of_thought=chain_of_thought,
        )

        text_response.additional_properties = d
        return text_response

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
