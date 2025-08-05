from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TensorMetadata(_message.Message):
    __slots__ = ("shape", "dtype", "compressed", "original_dtype")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_DTYPE_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    compressed: bool
    original_dtype: str
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., compressed: bool = ..., original_dtype: _Optional[str] = ...) -> None: ...

class LayerRequest(_message.Message):
    __slots__ = ("request_id", "input_tensor", "layer_indices", "metadata", "context")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    LAYER_INDICES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    input_tensor: bytes
    layer_indices: _containers.RepeatedScalarFieldContainer[int]
    metadata: TensorMetadata
    context: _containers.ScalarMap[str, str]
    def __init__(self, request_id: _Optional[str] = ..., input_tensor: _Optional[bytes] = ..., layer_indices: _Optional[_Iterable[int]] = ..., metadata: _Optional[_Union[TensorMetadata, _Mapping]] = ..., context: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LayerResponse(_message.Message):
    __slots__ = ("request_id", "output_tensor", "metadata", "processing_time_ms", "device_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    output_tensor: bytes
    metadata: TensorMetadata
    processing_time_ms: float
    device_id: str
    def __init__(self, request_id: _Optional[str] = ..., output_tensor: _Optional[bytes] = ..., metadata: _Optional[_Union[TensorMetadata, _Mapping]] = ..., processing_time_ms: _Optional[float] = ..., device_id: _Optional[str] = ...) -> None: ...

class HealthStatus(_message.Message):
    __slots__ = ("healthy", "device_id", "timestamp", "details")
    class DetailsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    device_id: str
    timestamp: int
    details: _containers.ScalarMap[str, str]
    def __init__(self, healthy: bool = ..., device_id: _Optional[str] = ..., timestamp: _Optional[int] = ..., details: _Optional[_Mapping[str, str]] = ...) -> None: ...

class DeviceInfo(_message.Message):
    __slots__ = ("device_id", "hostname", "rank", "role", "assigned_layers", "capabilities", "gpu_utilization", "memory_usage_gb")
    class CapabilitiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_LAYERS_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    GPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_GB_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    hostname: str
    rank: int
    role: str
    assigned_layers: _containers.RepeatedScalarFieldContainer[int]
    capabilities: _containers.ScalarMap[str, str]
    gpu_utilization: float
    memory_usage_gb: float
    def __init__(self, device_id: _Optional[str] = ..., hostname: _Optional[str] = ..., rank: _Optional[int] = ..., role: _Optional[str] = ..., assigned_layers: _Optional[_Iterable[int]] = ..., capabilities: _Optional[_Mapping[str, str]] = ..., gpu_utilization: _Optional[float] = ..., memory_usage_gb: _Optional[float] = ...) -> None: ...

class TensorTransfer(_message.Message):
    __slots__ = ("tensor_id", "shape", "data", "dtype", "source_rank", "dest_rank")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RANK_FIELD_NUMBER: _ClassVar[int]
    DEST_RANK_FIELD_NUMBER: _ClassVar[int]
    tensor_id: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    dtype: str
    source_rank: int
    dest_rank: int
    def __init__(self, tensor_id: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ..., dtype: _Optional[str] = ..., source_rank: _Optional[int] = ..., dest_rank: _Optional[int] = ...) -> None: ...

class TransferResponse(_message.Message):
    __slots__ = ("success", "message", "transfer_time_ms")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    transfer_time_ms: float
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., transfer_time_ms: _Optional[float] = ...) -> None: ...
