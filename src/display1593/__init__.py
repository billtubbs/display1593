__all__ = ["Display1593"]


def __getattr__(name):
    # Lazy so that importing lightweight submodules (e.g. image_conversion)
    # doesn't pull in the hardware-only deps (serial, numba, serial_comm)
    # needed by Display1593 itself.
    if name == "Display1593":
        from .display1593 import Display1593

        return Display1593
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
