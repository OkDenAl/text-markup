from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PredictionsRequest(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class PredictionResponce(_message.Message):
    __slots__ = ("token_list", "token_labels")
    TOKEN_LIST_FIELD_NUMBER: _ClassVar[int]
    TOKEN_LABELS_FIELD_NUMBER: _ClassVar[int]
    token_list: _containers.RepeatedScalarFieldContainer[str]
    token_labels: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, token_list: _Optional[_Iterable[str]] = ..., token_labels: _Optional[_Iterable[str]] = ...) -> None: ...
