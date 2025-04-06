import struct
import SlimeVRServer_Simple.UDP.UDPPackage as UDPPackage


def parse_packet(data: bytes, server_socket, addr):
    packet_id = struct.unpack_from(">I", data, 0)[0]
    if packet_id == 3:
        response = build_response()
        server_socket.sendto(response, addr)
        return packet_id, None
    elif packet_id == 4:
        x, y, z = UDPPackage.parse_acceleration(data)
        return packet_id, [x,y,z]

    elif packet_id == 17:
        quaternion = UDPPackage.parse_rotation(data)
        return packet_id, quaternion
    else:
        return packet_id,None


def build_response():
    #Initial verification information
    base_data = bytes([0x03]) + b'Hey OVR =D 5'
    padding = bytes(64 - len(base_data))
    return base_data + padding
