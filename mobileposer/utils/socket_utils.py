import socket
import time
import numpy as np

from mobileposer.constants import OUT_UDP_IP, OUT_UDP_PORT


def init_sockets(host, ports):
    sockets = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sockets.append(sock)
        print(f"Got the {port} socket to bind!")
    return sockets


def send_and_save_data(socket, virtual_acc, virtual_ori, save=True):
    acc, ori = [], []
    for _id in range(5):
        acc.append(virtual_acc[_id])
        ori.append(virtual_ori[_id][[3, 0, 1, 2]])

    a = np.array(acc)
    o = np.array(ori)

    s = ','.join(['%g' % v for v in a.flatten()]) + '#' + \
        ','.join(['%g' % v for v in o.flatten()]) + '$'

    # save the string with a unix_timestamp 
    unix_time = str(time.time())

    sensor_bytes = bytes(s, encoding="utf8")
    socket.sendto(sensor_bytes, (OUT_UDP_IP, OUT_UDP_PORT))