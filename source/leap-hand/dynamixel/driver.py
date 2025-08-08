from collections.abc import Sequence
import logging
from multiprocessing import Event
from threading import Lock
from threading import Thread
import time

from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS
from dynamixel_sdk.robotis_def import DXL_HIBYTE
from dynamixel_sdk.robotis_def import DXL_HIWORD
from dynamixel_sdk.robotis_def import DXL_LOBYTE
from dynamixel_sdk.robotis_def import DXL_LOWORD
import numpy as np

from .constants import ADDR_GOAL_POSITION
from .constants import ADDR_PRESENT_CURRENT
from .constants import ADDR_PRESENT_POS_VEL_CUR
from .constants import ADDR_PRESENT_POSITION
from .constants import ADDR_PRESENT_VELOCITY
from .constants import ADDR_TORQUE_ENABLE
from .constants import LEN_GOAL_POSITION
from .constants import LEN_PRESENT_CURRENT
from .constants import LEN_PRESENT_POS_VEL_CUR
from .constants import LEN_PRESENT_POSITION
from .constants import LEN_PRESENT_VELOCITY
from .protocol import DynamixelDriverProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_POS_SCALE = 2.0 * np.pi / 4096  # 0.088 degrees per unit
DEFAULT_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0  # 0.229 rpm
DEFAULT_CUR_SCALE = 1.34


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self,
        servo_ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 57600,
        pos_scale: float = DEFAULT_POS_SCALE,
        vel_scale: float = DEFAULT_VEL_SCALE,
        cur_scale: float = DEFAULT_CUR_SCALE,
        reading_interval: float = 0.001,
        reading_retries: int = 5,
    ):
        self.servo_ids = servo_ids
        self.port = port
        self.baud_rate = baud_rate
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.cur_scale = cur_scale
        self.reading_interval = reading_interval
        self.reading_retries = reading_retries

        self._port_handler = PortHandler(port)
        self._packet_handler = PacketHandler(2.0)

        self._joint_positions = np.zeros(len(servo_ids), dtype=np.int32)
        self._joint_velocities = np.zeros(len(servo_ids), dtype=np.int32)
        self._joint_currents = np.zeros(len(servo_ids), dtype=np.int32)
        self._lock = Lock()

        # Reader for joint positions, velocities, and currents
        self._group_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_POS_VEL_CUR,
            LEN_PRESENT_POS_VEL_CUR,
        )
        # Writer for goal positions
        self._group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # Open the port and set the baud rate
        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open port {port}")

        if not self._port_handler.setBaudRate(baud_rate):
            raise RuntimeError(f"Failed to set baud rate {baud_rate}")

        for servo_id in servo_ids:
            if not self._group_sync_read.addParam(servo_id):
                raise RuntimeError(f"Failed to add servo ID {servo_id} to sync read group")

        # Enable torque for each Dynamixel servo
        self._torque_enabled = False
        try:
            self.set_torque_mode(enable=self._torque_enabled)
        except Exception as e:
            logger.error(f"port: {port}, {e}")

        self._stop_thread = Event()
        self._start_reading_thread()

    def _reading_loop(self):
        retries = self.reading_retries
        while not self._stop_thread.is_set():
            time.sleep(self.reading_interval)
            with self._lock:
                dxl_comm_result = self._group_sync_read.fastSyncRead()
                if dxl_comm_result != COMM_SUCCESS:
                    retries -= 1
                    if retries <= 0:
                        logger.warning(
                            f"Failed to read data from Dynamixel servos after {self.reading_retries - retries} retries, data may be delayed or unavailable."
                        )
                    continue
                retries = self.reading_retries

                positions = np.zeros_like(self._joint_positions)
                velocities = np.zeros_like(self._joint_velocities)
                currents = np.zeros_like(self._joint_currents)

                for i, servo_id in enumerate(self.servo_ids):
                    if self._group_sync_read.isAvailable(servo_id, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR):
                        positions[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                            )
                        )
                        velocities[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                            )
                        )
                        currents[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
                            )
                        )

                self._joint_positions = positions
                self._joint_velocities = velocities
                self._joint_currents = currents

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._reading_loop, daemon=True)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, *, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        with self._lock:
            for dxl_id in self.servo_ids:
                dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                    self._port_handler, dxl_id, ADDR_TORQUE_ENABLE, int(enable)
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(f"Failed to set torque mode for Dynamixel with ID {dxl_id}")

        self._torque_enabled = enable

    def get_joint_positions(self):
        return self._joint_positions.copy() * self.pos_scale

    def set_joint_positions(self, joint_positions: Sequence[float]):
        self._group_sync_write.clearParam()
        assert len(joint_positions) == len(self.servo_ids), "Length of joint_positions must match number of servos"
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint positions")

        for servo_id, position in zip(self.servo_ids, joint_positions, strict=True):
            position_value = int(position / self.pos_scale) 
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            add_param_result = self._group_sync_write.addParam(servo_id, param_goal_position)
            if not add_param_result:
                raise RuntimeError(f"Failed to set joint angle for Dynamixel with ID {servo_id}")

        comm_result = self._group_sync_write.txPacket()
        if comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to sync write goal position")

        self._group_sync_write.clearParam()

    def get_joint_currents(self):
        return self._joint_currents.copy() * self.cur_scale

    def get_joint_velocities(self):
        return self._joint_velocities.copy() * self.vel_scale

    def close(self):
        self._stop_thread.set()
        self._reading_thread.join()
        self._port_handler.closePort()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    driver = DynamixelDriver(
        servo_ids=list(range(16)),
        port="/dev/cu.usbserial-FTA2U4SR",
        baud_rate=4_000_000,
    )
    try:
        driver.set_torque_mode(enable=True)
        driver.set_joint_positions(
            np.array(
                [
                    3.1415927410125732,
                    3.1415927410125732,
                    4.71238899230957,
                    4.71238899230957,
                    0.0,
                    3.1415927410125732,
                    3.1415927410125732,
                    0.0,
                    3.1415927410125732,
                    3.1415927410125732,
                    3.1415927410125732,
                    0.0,
                    4.71238899230957,
                    3.1415927410125732,
                    3.1415927410125732,
                    3.1415927410125732,
                ]
            )
        )

        while True:
            time.sleep(1)
            print(f"Joint positions: {driver.get_joint_positions()}")
    finally:
        driver.close()
