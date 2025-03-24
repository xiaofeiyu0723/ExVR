import struct


def parse_acceleration(data: bytes) -> tuple[float, float, float]:
    start_byte = 12
    vector_bytes = data[start_byte: start_byte + 12]
    if len(vector_bytes) != 12:
        raise ValueError("Invalid Vector3 data length")
    x, y, z = struct.unpack('>3f', vector_bytes)
    return x, y, z


def parse_rotation(data: bytes):
    start_byte = 14
    float_bytes = data[start_byte: start_byte + 16]
    x, y, z, w = struct.unpack('>4f', float_bytes)
    return [x, y, z, w]
