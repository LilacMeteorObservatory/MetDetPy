"""Photo device selection and Windows DML adapter discovery (no model loading).

DML device_id uses IDXGIFactory::EnumAdapters order, not performance order.
Native calls are isolated here and loaded only on Windows.
"""
import ctypes as ct
import sys
import uuid
from dataclasses import dataclass
from typing import Optional



@dataclass
class DMLAdapter:
    index: int
    name: str
    luid: tuple[int, int]
    vendor_id: int = 0
    device_id: int = 0
    software: bool = False
    physical_ids: tuple[str, ...] = ()
    identity_error: str = ""
    duplicate_of: Optional[int] = None

    @property
    def key(self):
        return f"dml:{self.index}"


def same_adapter(a, b):
    return (a.luid == b.luid
            or bool(a.physical_ids and b.physical_ids
                    and set(a.physical_ids) == set(b.physical_ids)))


def classify_adapters(adapters):
    """Mark aliases without collapsing distinct same-model GPUs or renumbering."""
    representatives = []
    for adapter in sorted(adapters, key=lambda a: a.index):
        adapter.duplicate_of = None
        if adapter.software:
            continue
        previous = next(
            (a for a in representatives if same_adapter(a, adapter)), None)
        if previous is not None:
            adapter.duplicate_of = previous.index
        else:
            representatives.append(adapter)
    return adapters


class _GUID(ct.Structure):
    _fields_ = [("data", ct.c_ubyte * 16)]

    def __init__(self, value):
        super().__init__()
        self.data[:] = uuid.UUID(value).bytes_le


class _LUID(ct.Structure):
    _fields_ = [("low", ct.c_uint32), ("high", ct.c_int32)]


class _AdapterDesc1(ct.Structure):
    _fields_ = [("name", ct.c_wchar * 128), ("vendor", ct.c_uint32),
                ("device", ct.c_uint32), ("subsystem", ct.c_uint32),
                ("revision", ct.c_uint32), ("video_memory", ct.c_size_t),
                ("system_memory", ct.c_size_t), ("shared_memory", ct.c_size_t),
                ("luid", _LUID), ("flags", ct.c_uint32)]


def _method(obj, slot, result, *arguments):
    vtable = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(result, ct.c_void_p, *arguments)(vtable[slot])


def _check(status, operation):
    if status < 0:
        raise OSError(f"{operation}: 0x{status & 0xffffffff:08x}")


def _release(obj):
    if obj:
        _method(obj, 2, ct.c_uint32)(obj)


def _physical_ids(luid):
    """Query hardware PnP registry keys for every physical member of an LDA.

    A hardware key identifies a device instance; PCI vendor/device IDs do not.
    KMTQAITYPE_PHYSICALADAPTERCOUNT=30, PHYSICALADAPTERPNPKEY=41.
    """

    class Open(ct.Structure):
        _fields_ = [("luid", _LUID), ("handle", ct.c_uint32)]

    class Query(ct.Structure):
        _fields_ = [("handle", ct.c_uint32), ("type", ct.c_int32),
                    ("data", ct.c_void_p), ("size", ct.c_uint32)]

    class PnP(ct.Structure):
        _fields_ = [("physical_index", ct.c_uint32), ("key_type", ct.c_int32),
                    ("destination", ct.c_void_p),
                    ("characters", ct.POINTER(ct.c_uint32))]

    class Close(ct.Structure):
        _fields_ = [("handle", ct.c_uint32)]

    gdi = ct.WinDLL("gdi32")
    open_fn, query_fn, close_fn = (gdi.D3DKMTOpenAdapterFromLuid,
                                   gdi.D3DKMTQueryAdapterInfo,
                                   gdi.D3DKMTCloseAdapter)
    for fn, struct in ((open_fn, Open), (query_fn, Query), (close_fn, Close)):
        fn.argtypes, fn.restype = [ct.POINTER(struct)], ct.c_int32
    opened = Open(luid, 0)
    _check(open_fn(ct.byref(opened)), "D3DKMTOpenAdapterFromLuid")
    try:
        count = ct.c_uint32()
        query = Query(opened.handle, 30, ct.addressof(count), ct.sizeof(count))
        _check(query_fn(ct.byref(query)), "physical adapter count")
        if not 0 < count.value <= 64:
            raise OSError(f"Invalid physical adapter count: {count.value}")
        keys = []
        for index in range(count.value):
            buffer = ct.create_unicode_buffer(4096)
            length = ct.c_uint32(len(buffer))
            pnp = PnP(index, 1, ct.addressof(buffer),
                      ct.pointer(length))  # HARDWARE=1
            query = Query(opened.handle, 41, ct.addressof(pnp), ct.sizeof(pnp))
            _check(query_fn(ct.byref(query)), "physical adapter PnP key")
            if not buffer.value:
                raise OSError("Empty physical adapter PnP key")
            keys.append(buffer.value.casefold())
        return tuple(sorted(set(keys)))
    finally:
        close_fn(ct.byref(Close(opened.handle)))


def discover_dml_adapters():
    if sys.platform != "win32":
        return []
    factory = ct.c_void_p()
    iid = _GUID("7b7166ec-21c7-44ae-b21a-c9ae321ae369")  # IDXGIFactory
    create = ct.WinDLL("dxgi").CreateDXGIFactory
    create.argtypes = [ct.POINTER(_GUID), ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_int32
    _check(create(ct.byref(iid), ct.byref(factory)), "CreateDXGIFactory")
    adapters = []
    try:
        index = 0
        while True:
            adapter, adapter1 = ct.c_void_p(), ct.c_void_p()
            status = _method(factory, 7, ct.c_int32, ct.c_uint32,
                             ct.POINTER(ct.c_void_p))(factory, index,
                                                      ct.byref(adapter))
            # IDXGIFactory::EnumAdapters is slot 7 (slot 6 is GetParent).
            if status & 0xffffffff == 0x887a0002:  # DXGI_ERROR_NOT_FOUND
                break
            _check(status, "EnumAdapters")
            try:
                iid = _GUID("29038f61-3839-4626-91fd-086879011a05")
                _check(
                    _method(adapter, 0, ct.c_int32, ct.POINTER(_GUID),
                            ct.POINTER(ct.c_void_p))(adapter, ct.byref(iid),
                                                     ct.byref(adapter1)),
                    "IDXGIAdapter1")
                desc = _AdapterDesc1()
                _check(
                    _method(adapter1, 10, ct.c_int32,
                            ct.POINTER(_AdapterDesc1))(adapter1,
                                                       ct.byref(desc)),
                    "GetDesc1")
                item = DMLAdapter(index, desc.name,
                                  (desc.luid.low, desc.luid.high), desc.vendor,
                                  desc.device, bool(desc.flags & 2))
                if not item.software:
                    try:
                        item.physical_ids = _physical_ids(desc.luid)
                    except OSError as error:
                        item.identity_error = str(error)
                adapters.append(item)
            finally:
                _release(adapter1)
                _release(adapter)
            index += 1
    finally:
        _release(factory)
    return classify_adapters(adapters)


def describe_adapter(adapter: DMLAdapter):
    reason = ("excluded: software"
              if adapter.software else f"alias of dml:{adapter.duplicate_of}"
              if adapter.duplicate_of is not None else
              "candidate (model availability checked at initialization)")
    physical = ", ".join(adapter.physical_ids) or "unknown"
    return (f"{adapter.key}: {adapter.name}; LUID={adapter.luid}; "
            f"physical={physical}; {reason}" +
            (f"; identity query failed: {adapter.identity_error}"
             if adapter.identity_error else ""))


def parse_photo_devices(value: str):
    # Keep native device discovery independent of the inference model module.
    from .model import validate_provider_key

    keys = value.split(",")
    if not keys or any(not key for key in keys):
        raise ValueError("Device list must not contain empty entries.")
    for key in keys:
        if key != "gpu":
            validate_provider_key(key)
    if "dml" in keys and any(key.startswith("dml:") for key in keys):
        raise ValueError("Do not mix automatic 'dml' with explicit dml:N devices.")
    if ("gpu" in keys or "default"
            in keys) and keys not in (["gpu"], ["gpu", "cpu"], ["default"]):
        raise ValueError(
            "Use 'gpu', 'gpu,cpu', 'default', or explicit device names.")
    return list(dict.fromkeys(keys))
