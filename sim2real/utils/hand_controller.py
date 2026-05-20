import threading
import time


DEFAULT_INSPIRE_OPEN_RAW = [900, 900, 900, 900, 900, 600]
DEFAULT_INSPIRE_CLOSE_RAW = [350, 350, 350, 350, 420, 600]
INSPIRE_CMD_TOPIC = "rt/inspire/cmd"


def _log(logger, level, message):
    if logger is None:
        print(message)
        return
    getattr(logger, level, logger.info)(message)


def _parse_six_values(value, default, name):
    if value is None:
        value = default
    if isinstance(value, str):
        text = value.strip().strip("[]()")
        value = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    values = [float(item) for item in value]
    if len(values) != 6:
        raise ValueError(f"{name} must contain exactly 6 values, got {len(values)}")
    return values


def normalize_inspire_q(value, default, name):
    values = _parse_six_values(value, default, name)
    if max(values) > 1.5:
        values = [item / 1000.0 for item in values]
    for item in values:
        if item < 0.0 or item > 1.0:
            raise ValueError(f"{name} normalized values must be in [0, 1], got {values}")
    return values


def inspire_right_hand_q12(right_q):
    right_q = [float(item) for item in right_q]
    if len(right_q) != 6:
        raise ValueError(f"right hand q must contain 6 values, got {len(right_q)}")
    return right_q + [1.0] * 6


class HandController:
    def open(self, force=False):
        raise NotImplementedError

    def close(self, force=False):
        raise NotImplementedError

    def hold(self):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError


class MockSimHandController(HandController):
    """Preserve the existing MuJoCo hand path by writing the sim control file."""

    def __init__(self, sim_control_writer=None, extra_payload_fn=None, logger=None):
        self.sim_control_writer = sim_control_writer
        self.extra_payload_fn = extra_payload_fn
        self.logger = logger
        self._state = None
        _log(self.logger, "info", "[HAND] backend=mock")

    def _send_state(self, state, force=False):
        if not force and self._state == state:
            return
        self._state = state
        payload = {
            "right_hand_state": state,
            "timestamp": time.time(),
        }
        if self.extra_payload_fn is not None:
            payload.update(self.extra_payload_fn() or {})
        if self.sim_control_writer is not None:
            self.sim_control_writer(payload)
        _log(self.logger, "info", f"[HAND] mock right hand {state}")

    def open(self, force=False):
        self._send_state("open", force=force)

    def close(self, force=False):
        self._send_state("close", force=force)

    def hold(self):
        # The sim keeps the last written right_hand_state target.
        return

    def shutdown(self):
        return


class InspireDDSRightHandController(HandController):
    """Right-hand RH56DF3 controller through Unitree Inspire DDS bridge."""

    def __init__(self, config, logger=None, initialize_channel_factory=False):
        self.config = config
        self.logger = logger
        self.hand_side = str(config.get("inspire_hand_side", "right")).strip().lower()
        if self.hand_side != "right":
            raise ValueError("Only inspire_hand_side=right is supported in this RH56DF3 deployment path")
        self.open_q = normalize_inspire_q(
            config.get("inspire_open_raw", DEFAULT_INSPIRE_OPEN_RAW),
            DEFAULT_INSPIRE_OPEN_RAW,
            "inspire_open_raw",
        )
        self.close_q = normalize_inspire_q(
            config.get("inspire_close_raw", DEFAULT_INSPIRE_CLOSE_RAW),
            DEFAULT_INSPIRE_CLOSE_RAW,
            "inspire_close_raw",
        )
        self.publish_rate_hz = float(config.get("inspire_publish_rate_hz", 20.0))
        self.publish_duration_sec = float(config.get("inspire_publish_duration_sec", 1.0))
        self.dds_channel = int(config.get("inspire_dds_channel", 0))
        self.network_interface = config.get("inspire_network_interface", None) or None
        self.topic = str(config.get("inspire_cmd_topic", INSPIRE_CMD_TOPIC))
        self._state = None
        self._last_q12 = None
        self._generation = 0
        self._shutdown = False
        self._lock = threading.Lock()
        self._thread = None

        self._init_dds(initialize_channel_factory=initialize_channel_factory)
        _log(self.logger, "info", "[HAND] backend=inspire_dds")
        _log(self.logger, "info", f"[HAND] side=right")
        _log(self.logger, "info", f"[HAND] open_q normalized={self.open_q}")
        _log(self.logger, "info", f"[HAND] close_q normalized={self.close_q}")
        _log(self.logger, "info", f"[HAND] DDS topic={self.topic}")
        _log(
            self.logger,
            "info",
            f"[HAND] DDS channel={self.dds_channel}, interface={self.network_interface or '<default>'}",
        )

    @property
    def last_command_q12(self):
        return None if self._last_q12 is None else list(self._last_q12)

    def _init_dds(self, initialize_channel_factory):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
            from unitree_sdk2py.idl.default import MotorCmd_, MotorCmds_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_ as MotorCmdsType
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("unitree_sdk2py"):
                raise RuntimeError(
                    "hand_backend=inspire_dds requires unitree_sdk2py, but it is not installed "
                    "in the current Python environment. Activate the fcreal conda environment "
                    "or run with PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python."
                ) from exc
            raise
        except ImportError as exc:
            raise RuntimeError(
                "hand_backend=inspire_dds could not import MotorCmds_/MotorCmd_ from unitree_sdk2py. "
                "Check that the fcreal environment uses the Unitree SDK2 Python package with "
                "unitree_go::msg::dds_::MotorCmds_ support."
            ) from exc

        if initialize_channel_factory:
            if self.network_interface:
                ChannelFactoryInitialize(self.dds_channel, self.network_interface)
            else:
                ChannelFactoryInitialize(self.dds_channel)
        self._motor_cmd_cls = MotorCmd_
        self._motor_cmds_cls = MotorCmds_
        self._publisher = ChannelPublisher(self.topic, MotorCmdsType)
        self._publisher.Init()

    def _make_msg(self, q12):
        cmds = [
            self._motor_cmd_cls(0, float(q), 0.0, 0.0, 0.0, 0.0, [0, 0, 0])
            for q in q12
        ]
        return self._motor_cmds_cls(cmds)

    def _publish_once(self, q12):
        self._publisher.Write(self._make_msg(q12))
        self._last_q12 = list(q12)

    def _publish_for_duration(self, q12, generation):
        period = 1.0 / max(self.publish_rate_hz, 1e-6)
        deadline = time.monotonic() + max(0.0, self.publish_duration_sec)
        next_t = time.monotonic() + period
        while time.monotonic() < deadline:
            with self._lock:
                if self._shutdown or generation != self._generation:
                    return
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            with self._lock:
                if self._shutdown or generation != self._generation:
                    return
            self._publish_once(q12)
            next_t += period

    def _command(self, state, right_q, force=False):
        with self._lock:
            if not force and self._state == state:
                return
            self._state = state
            self._generation += 1
            generation = self._generation
            q12 = inspire_right_hand_q12(right_q)
            self._publish_once(q12)
            self._thread = threading.Thread(
                target=self._publish_for_duration,
                args=(q12, generation),
                daemon=True,
            )
            self._thread.start()
        _log(self.logger, "info", f"[HAND] inspire_dds right hand {state}: q12={q12}")

    def open(self, force=False):
        self._command("open", self.open_q, force=force)

    def close(self, force=False):
        self._command("close", self.close_q, force=force)

    def hold(self):
        if self._state == "close":
            self._command("hold", self.close_q, force=True)

    def shutdown(self):
        with self._lock:
            self._shutdown = True
            self._generation += 1
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)


def create_hand_controller(
    config,
    sim_control_writer=None,
    extra_payload_fn=None,
    logger=None,
    initialize_dds=False,
):
    backend = str(config.get("hand_backend", "mock")).strip().lower()
    if backend in ("mock", "sim"):
        return MockSimHandController(
            sim_control_writer=sim_control_writer,
            extra_payload_fn=extra_payload_fn,
            logger=logger,
        )
    if backend == "inspire_dds":
        return InspireDDSRightHandController(
            config,
            logger=logger,
            initialize_channel_factory=initialize_dds,
        )
    raise ValueError(f"Unsupported hand_backend={backend!r}; expected mock or inspire_dds")
