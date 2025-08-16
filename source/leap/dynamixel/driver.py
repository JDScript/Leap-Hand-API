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

from .constants import ADDR_GOAL_CURRENT
from .constants import ADDR_GOAL_POSITION
from .constants import ADDR_HOMING_OFFSET
from .constants import ADDR_OPERATING_MODE
from .constants import ADDR_POSITION_D_GAIN
from .constants import ADDR_POSITION_P_GAIN
from .constants import ADDR_PRESENT_CURRENT
from .constants import ADDR_PRESENT_POS_VEL_CUR
from .constants import ADDR_PRESENT_POSITION
from .constants import ADDR_PRESENT_VELOCITY
from .constants import ADDR_TORQUE_ENABLE
from .constants import ADDR_VELOCITY_P_GAIN
from .constants import SIZE_GOAL_POSITION
from .constants import SIZE_PRESENT_CURRENT
from .constants import SIZE_PRESENT_POS_VEL_CUR
from .constants import SIZE_PRESENT_POSITION
from .constants import SIZE_PRESENT_VELOCITY
from .protocol import DynamixelDriverProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_POS_SCALE = 2.0 * np.pi / 4096  # 0.088 degrees per unit
DEFAULT_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0  # 0.229 rpm
DEFAULT_CUR_SCALE = 1.34


class DynamixelDriver(DynamixelDriverProtocol):
    def _write_with_retry(
        self, write_func, *args, retries: int = 3, retry_interval: float = 0.02, err_msg: str = "", **kwargs
    ):
        attempt = 0
        while attempt <= retries:
            result = write_func(*args, **kwargs)
            # write1ByteTxRx, write4ByteTxRx等返回 (comm_result, dxl_error)
            if isinstance(result, tuple) and len(result) == 2:
                comm_result, dxl_error = result
            else:
                comm_result, dxl_error = result, 0
            if comm_result == COMM_SUCCESS and dxl_error == 0:
                return True
            logger.warning(
                f"Attempt {attempt + 1}/{retries + 1}: {err_msg} (COMM_RESULT={comm_result}, ERROR={dxl_error})"
            )
            attempt += 1
            time.sleep(retry_interval)
        raise RuntimeError(f"{err_msg} after {retries + 1} attempts")

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
            SIZE_PRESENT_POS_VEL_CUR,
        )
        # Writer for goal positions
        self._group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            ADDR_GOAL_POSITION,
            SIZE_GOAL_POSITION,
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
                    if self._group_sync_read.isAvailable(servo_id, ADDR_PRESENT_POS_VEL_CUR, SIZE_PRESENT_POS_VEL_CUR):
                        positions[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_POSITION, SIZE_PRESENT_POSITION)
                            )
                        )
                        velocities[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_VELOCITY, SIZE_PRESENT_VELOCITY)
                            )
                        )
                        currents[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_CURRENT, SIZE_PRESENT_CURRENT)
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

    def set_operation_mode(self, mode: int):
        """Set the operation mode for the Dynamixel servos.

        Args:
            mode (int): The operation mode to set.
        """
        with self._lock:
            for dxl_id in self.servo_ids:
                dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                    self._port_handler, dxl_id, ADDR_OPERATING_MODE, mode
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(f"Failed to set operation mode for Dynamixel with ID {dxl_id}")

    def set_p_gain(self, gains: list[float], retries: int = 3, retry_interval: float = 0.02):
        """Set the P gain for the Dynamixel servos, with retries and better error logging."""
        with self._lock:
            for dxl_id, gain in zip(self.servo_ids, gains, strict=False):
                self._write_with_retry(
                    self._packet_handler.write2ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_POSITION_P_GAIN,
                    int(gain),
                    retries=retries,
                    retry_interval=retry_interval,
                    err_msg=f"Failed to set P gain for Dynamixel ID {dxl_id}",
                )

    def set_velocity_p_gain(self, gains: list[float], retries: int = 3, retry_interval: float = 0.02):
        """Set the velocity P gain for the Dynamixel servos, with retries and better error logging."""
        with self._lock:
            for dxl_id, gain in zip(self.servo_ids, gains, strict=False):
                self._write_with_retry(
                    self._packet_handler.write2ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_VELOCITY_P_GAIN,
                    int(gain),
                    retries=retries,
                    retry_interval=retry_interval,
                    err_msg=f"Failed to set velocity P gain for Dynamixel ID {dxl_id}",
                )

    def set_position_d_gain(self, gains: list[float], retries: int = 3, retry_interval: float = 0.02):
        """Set the position D gain for the Dynamixel servos, with retries and better error logging."""
        with self._lock:
            for dxl_id, gain in zip(self.servo_ids, gains, strict=False):
                self._write_with_retry(
                    self._packet_handler.write2ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_POSITION_D_GAIN,
                    int(gain),
                    retries=retries,
                    retry_interval=retry_interval,
                    err_msg=f"Failed to set position D gain for Dynamixel ID {dxl_id}",
                )

    def set_goal_current(self, currents: list[float], retries: int = 3, retry_interval: float = 0.02):
        """Set the goal current for the Dynamixel servos, with retries and better error logging."""
        with self._lock:
            for dxl_id, current in zip(self.servo_ids, currents, strict=False):
                self._write_with_retry(
                    self._packet_handler.write2ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_GOAL_CURRENT,
                    int(current),
                    retries=retries,
                    retry_interval=retry_interval,
                    err_msg=f"Failed to set goal current for Dynamixel ID {dxl_id}",
                )

    def get_p_gain(self) -> dict[int, float]:
        with self._lock:
            p_gains = {}
            for dxl_id in self.servo_ids:
                data, dxl_comm_result, dxl_error = self._packet_handler.read2ByteTxRx(
                    self._port_handler, dxl_id, ADDR_POSITION_P_GAIN
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(f"Failed to get P gain for Dynamixel with ID {dxl_id}")
                p_gains[dxl_id] = float(data) * self.pos_scale
            return p_gains

    def get_operation_mode(self) -> dict[int, int]:
        with self._lock:
            modes = {}
            for dxl_id in self.servo_ids:
                data, dxl_comm_result, dxl_error = self._packet_handler.read1ByteTxRx(
                    self._port_handler, dxl_id, ADDR_OPERATING_MODE
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(f"Failed to get operation mode for Dynamixel with ID {dxl_id}")
                modes[dxl_id] = data
        return modes

    def set_homing_offset(self, offsets: list[int], retries: int = 3, retry_interval: float = 0.02):
        """Set the homing offset for the Dynamixel servos, with retries and better error logging."""
        with self._lock:
            for dxl_id, offset in zip(self.servo_ids, offsets, strict=False):
                self._write_with_retry(
                    self._packet_handler.write4ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_HOMING_OFFSET,
                    int(offset),
                    retries=retries,
                    retry_interval=retry_interval,
                    err_msg=f"Failed to set homing offset for Dynamixel ID {dxl_id}",
                )

    def set_torque_mode(self, *, enable: bool, retries: int = 3, retry_interval: float = 0.02):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        with self._lock:
            for dxl_id in self.servo_ids:
                self._write_with_retry(
                    self._packet_handler.write1ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_TORQUE_ENABLE,
                    int(enable),
                    retries=retries,
                    retry_interval=retry_interval,
                    err_msg=f"Failed to set torque mode for Dynamixel ID {dxl_id}",
                )

        self._torque_enabled = enable

    def get_joint_positions(self):
        return self._joint_positions.copy() * self.pos_scale

    def set_joint_positions(self, joint_positions: Sequence[float]):
        self._group_sync_write.clearParam()
        assert len(joint_positions) == len(self.servo_ids), "Length of joint_positions must match number of servos"
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint positions")

        error_ids = []
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
                error_ids.append(servo_id)

        if error_ids:
            logger.error(f"Failed to set joint positions for Dynamixel IDs: {error_ids}")

        with self._lock:
            comm_result = self._group_sync_write.txPacket()
            if comm_result != COMM_SUCCESS:
                self.handle_packet_result(comm_result, context="sync_write")

        self._group_sync_write.clearParam()

    def handle_packet_result(
        self,
        comm_result: int,
        dxl_error: int | None = None,
        dxl_id: int | None = None,
        context: str | None = None,
    ):
        """Handles the result from a communication request."""
        error_message = None
        if comm_result != COMM_SUCCESS:
            error_message = self._packet_handler.getTxRxResult(comm_result)
        elif dxl_error is not None:
            error_message = self._packet_handler.getRxPacketError(dxl_error)
        if error_message:
            if dxl_id is not None:
                error_message = f"[Motor ID: {dxl_id}] {error_message}"
            if context is not None:
                error_message = f"> {context}: {error_message}"
            logger.error(error_message)
            return False
        return True

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
