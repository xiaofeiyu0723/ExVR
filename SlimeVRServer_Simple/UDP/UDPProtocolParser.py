import struct
import SlimeVRServer_Simple.UDP.UDPPackage as UDPPackage
from SlimeVRServer_Simple.filter import AccelerometerProcessor


def parse_packet(data: bytes, server_socket, addr, accelerometer_processor: AccelerometerProcessor):
    packet_id = struct.unpack_from(">I", data, 0)[0]
    if packet_id == 3:
        response = build_response()
        server_socket.sendto(response, addr)
    elif packet_id == 4:
        x, y, z = UDPPackage.parse_acceleration(data)
        accelerometer_processor.add_acceleration_data((x, y, z))

    elif packet_id == 17:
        quaternion = UDPPackage.parse_rotation(data)
        accelerometer_processor.add_quaternion_data(quaternion)
    else:
        return {"error": f"Unknown package type: {packet_id}"}


def build_response():
    #Initial verification information
    base_data = bytes([0x03]) + b'Hey OVR =D 5'
    padding = bytes(64 - len(base_data))
    return base_data + padding
