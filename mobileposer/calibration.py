import os
import time
import socket
import threading
import numpy as np
from collections import defaultdict
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from queue import Queue
import select
from argparse import ArgumentParser

from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from numpy.linalg import inv

from mobileposer.config import *
from mobileposer.utils.sensor_utils import *
from mobileposer.utils.socket_utils import *
from mobileposer.utils.draw_utils import *
from mobileposer.visualizer import *


class PerformanceLogger:
    def __init__(self, num_devices, log_interval=1.0):
        self.num_devices = num_devices
        self.log_interval = log_interval
        self.counters = defaultdict(int)
        self.delay_sums = defaultdict(float)
        self.last_log_time = time.time()

    def update(self, device_id, delay):
        self.counters[device_id] += 1
        self.delay_sums[device_id] += delay

    def log(self):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self._log_performance()
            self._reset()
            self.last_log_time = current_time

    def _log_performance(self):
        time_diff = time.time() - self.last_log_time
        for device_id in range(self.num_devices):
            count = self.counters[device_id]
            total_delay = self.delay_sums[device_id]
            if count == 0 and total_delay == 0:
                continue

            frequency = count / time_diff if time_diff > 0 else 0
            average_delay = (total_delay / count) if count > 0 else 0

            print(f"Device_ID {device_id}: Frequency = {frequency:.2f} Hz, "
                  f"Average Delay = {average_delay*1000:.2f} ms")

    def _reset(self):
        self.counters.clear()
        self.delay_sums.clear()


class DataReceiver:
    def __init__(self, sockets, chunk_size):
        self.sockets = sockets
        self.chunk_size = chunk_size
        self.data_queue = Queue()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_data, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _receive_data(self):
        while self.running:
            try:
                readable, _, _ = select.select(self.sockets, [], [], 0.001)
                for sock in readable:
                    self._read_socket(sock)
            except Exception as e:
                print(f"Error in data receiver: {e}")

    def _read_socket(self, sock):
        while True:
            try:
                data, addr = sock.recvfrom(self.chunk_size, socket.MSG_DONTWAIT)
                receive_time = time.time()
                self.data_queue.put((data, addr, receive_time))
            except BlockingIOError:
                break  # No more data to read from this socket
            except Exception as e:
                print(f"Error reading from socket: {e}")
                break

    def get_data(self):
        return self.data_queue.get_nowait() if not self.data_queue.empty() else None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log", action='store_true')
    args = parser.parse_args()

    # Init socket and data handlers
    sockets = init_sockets(HOST, PORTS)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sensor_data = SensorData()

    # Initialize and start the data receiver
    data_receiver = DataReceiver(sockets, CHUNK)
    data_receiver.start()

    # Setup PyGame manager
    manager = PyGameManager(860, 860)

    # Initialize cubes for each device
    cubes = [Cube(device_id=i, position=DEVICE_POSITIONS[i]) for i in range(len(DEVICE_POSITIONS))]
    for cube in cubes:
        manager.add_cube(cube)

    # Initialize performance logger
    performance_logger = PerformanceLogger(num_devices=len(DEVICE_POSITIONS)) if args.log else None

    frames = 0
    prev_timestamp = 0
    curr_timestamp = 0
    glb_ori, glb_acc = None, None

    while True:
        continue_running, key_c_pressed = manager.handle_events()
        if not continue_running:
            break

        try:
            # Receive packet from server 
            data_packet = data_receiver.get_data()
            if data_packet is None:
                continue
            data, addr, receive_time = data_packet

            # Process received data
            vis_str, device_id, curr_acc, curr_ori, timestamps = process_data(data)
            curr_timestamp = sensor_data.update(device_id, curr_acc, curr_ori, timestamps)
            glb_ori, glb_acc = sensor2global(
                sensor_data.get_orientation(device_id),
                sensor_data.get_acceleration(device_id),
                sensor_data.calibration_quats,
                device_id
            )
            sensor_data.update_virtual(device_id, glb_acc, glb_ori)  # Update virtual data for display

            # Update cube orientation
            cubes[device_id].set_orientation(sensor_data.virtual_ori[device_id])

            # Calculate delay (for debugging)
            if performance_logger:
                delay = receive_time - timestamps[0]
                performance_logger.update(device_id, delay)

            time_diff = curr_timestamp - prev_timestamp
            if time_diff >= min_time_diff:
                # send data via socket to live demo
                send_and_save_data(send_sock, sensor_data.virtual_acc, sensor_data.virtual_ori)
                prev_timestamp = curr_timestamp

        except Exception as e:
            print("Exception encountered during data processing: ", e)
        except KeyboardInterrupt:
            data_receiver.stop()

        # Draw cubes using the PyGameManager
        manager.update(glb_acc)

        # Log frequency (FPS) and data receiving (delay)
        if performance_logger:
            performance_logger.log()
