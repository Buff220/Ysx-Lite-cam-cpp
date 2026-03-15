"""
YsxLite P2P camera viewer — Python / asyncio / OpenCV
Usage: python camera.py [broadcast_ip]   (default: 192.168.1.255)
"""

import asyncio
import struct
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# ─── Protocol constants ───────────────────────────────────────────────────────

class Cmd:
    LanSearch    = 61744
    P2PAlive     = 61920
    P2PAliveAck  = 61921
    P2pRdy       = 61762
    DrwAck       = 61905
    Drw          = 61904
    PunchPkt     = 61761
    HelloAck     = 61697
    Hello        = 61696
    P2pReq       = 61728
    LstReq       = 61799
    PunchTo      = 61760
    RlyTo        = 61698
    DevLgnAck    = 61713
    P2PReqAck    = 61729
    ListenReqAck = 61801
    RlyHelloAck  = 61808
    RlyHelloAck2 = 61809
    Close        = 61936

class CtrlCmd:
    ConnectUser     = 0x2010
    ConnectUserAck  = 0x2011
    StartVideo      = 0x1030
    StartVideoAck   = 0x1031
    VideoParamSet   = 0x1830
    VideoParamSetAck= 0x1831
    DevStatusAck    = 0x0811

CC_DEST = {
    CtrlCmd.ConnectUser: 0xFF00,
    CtrlCmd.StartVideo:  0,
}

FRAME_HEADER = bytes([0x55, 0xAA, 0x15, 0xA8])
JPEG_HEADER  = bytes([0xFF, 0xD8, 0xFF, 0xDB])

# ─── Crypto helpers ───────────────────────────────────────────────────────────

def xq_enc(buf: bytearray, length: int, rotate: int) -> None:
    tmp = bytearray(length)
    for i in range(length):
        b = buf[i]
        tmp[i] = b - 1 if b & 1 else b + 1
    buf[:length - rotate] = tmp[rotate:length]
    buf[length - rotate:length] = tmp[:rotate]

def xq_dec(buf: bytearray, length: int, rotate: int) -> None:
    tmp = bytearray(length)
    for i in range(length):
        b = buf[i]
        tmp[i] = b - 1 if b & 1 else b + 1
    buf[rotate:length] = tmp[:length - rotate]
    buf[:rotate] = tmp[length - rotate:length]

def u16_swap(x: int) -> int:
    return ((x & 0xFF00) >> 8) | ((x & 0x00FF) << 8)

# ─── Packet builders ──────────────────────────────────────────────────────────

def make_drw(session: "Session", command: int, data: bytes | None) -> bytes:
    TOKEN_LEN = 4
    HDR       = 16
    START_CMD = 0x110A

    payload_len = TOKEN_LEN
    enc_data: bytearray | None = None

    if data and len(data) > 4:
        enc_data = bytearray(data)
        xq_enc(enc_data, len(enc_data), 4)
        payload_len += len(enc_data)

    buf = bytearray(HDR + payload_len)
    struct.pack_into(">HH", buf, 0, Cmd.Drw, len(buf) - 4)
    buf[4] = 0xD1
    buf[5] = 0                             # channel
    struct.pack_into(">HHH", buf, 6,
                     session.outgoing_cmd_id, START_CMD, command)
    struct.pack_into(">HH", buf, 12,
                     u16_swap(payload_len), CC_DEST.get(command, 0))
    buf[16:20] = session.ticket
    if enc_data:
        buf[20:20 + len(enc_data)] = enc_data

    session.outgoing_cmd_id += 1
    return bytes(buf)

def make_lan_search() -> bytes:
    return struct.pack(">HH", Cmd.LanSearch, 0)

def make_p2p_rdy(dev: "Device") -> bytes:
    body = bytearray(20)
    _write_str(body, 0, dev.prefix)
    struct.pack_into(">Q", body, 4, dev.serial_u64)
    _write_str(body, 12, dev.suffix)
    hdr = struct.pack(">HH", Cmd.P2pRdy, 20)
    return hdr + bytes(body)

def make_p2p_alive()     -> bytes: return struct.pack(">HH", Cmd.P2PAlive, 0)
def make_p2p_alive_ack() -> bytes: return struct.pack(">HH", Cmd.P2PAliveAck, 0)

def make_drw_ack(data: bytes) -> bytes:
    pkt_id   = struct.unpack_from(">H", data, 6)[0]
    m_stream = data[5]
    return struct.pack(">HHBBHH", Cmd.DrwAck, 6, 0xD2, m_stream, 1, pkt_id)

def build_usr_chk(session: "Session") -> bytes:
    buf = bytearray(160)
    _write_str(buf,  0, "admin")
    _write_str(buf, 32, "admin")
    return make_drw(session, CtrlCmd.ConnectUser, bytes(buf))

def build_start_video(session: "Session") -> bytes:
    return make_drw(session, CtrlCmd.StartVideo, None)

def build_video_resolution(session: "Session", resol: int = 2) -> bytes:
    presets = {2: bytes([1, 0, 0, 0, 2, 0, 0, 0])}
    return make_drw(session, CtrlCmd.VideoParamSet, presets.get(resol, presets[2]))

# ─── String helpers ───────────────────────────────────────────────────────────

def _write_str(buf: bytearray, offset: int, s: str) -> None:
    for i, ch in enumerate(s):
        buf[offset + i] = ord(ch)

def _read_str(data: bytes, offset: int, length: int) -> str:
    raw = data[offset:offset + length]
    s   = raw.decode("latin-1")
    z   = s.find("\0")
    return s[:z] if z != -1 else s

# ─── Device info ─────────────────────────────────────────────────────────────

@dataclass
class Device:
    prefix:     str
    suffix:     str
    serial_u64: int

    @property
    def serial(self) -> str:
        return str(self.serial_u64)

    @property
    def dev_id(self) -> str:
        return self.prefix + self.serial + self.suffix

def parse_punch_pkt(data: bytes) -> Device:
    length      = struct.unpack_from(">H", data, 2)[0]
    prefix      = _read_str(data, 4, 4)
    serial_u64  = struct.unpack_from(">Q", data, 8)[0]
    suffix      = _read_str(data, 16, length - 16 + 4)
    return Device(prefix=prefix, suffix=suffix, serial_u64=serial_u64)

# ─── Session ──────────────────────────────────────────────────────────────────

@dataclass
class Session:
    dev:              Device
    dst_ip:           str
    dst_port:         int
    transport:        asyncio.BaseTransport = field(default=None, repr=False)
    outgoing_cmd_id:  int   = 0
    ticket:           bytes = b"\x00\x00\x00\x00"
    last_rx:          float = field(default_factory=time.monotonic)
    started:          bool  = False
    cur_image:        list  = field(default_factory=list)
    rcv_seq_id:       int   = 0
    frame_is_bad:     bool  = False
    unacked:          dict  = field(default_factory=dict)

    # callbacks set by caller
    on_frame:      callable = None
    on_disconnect: callable = None

    def send(self, pkt: bytes) -> None:
        if self.transport:
            self.transport.sendto(pkt, (self.dst_ip, self.dst_port))

    def close(self) -> None:
        self._disconnect()

    def _disconnect(self) -> None:
        if self.transport:
            try:
                self.transport.close()
            except Exception:
                pass
        print(f"[{self.dev.dev_id}] Disconnected")
        if self.on_disconnect:
            self.on_disconnect(self)

# ─── Frame reassembly ─────────────────────────────────────────────────────────

def deal_with_data(session: Session, data: bytes) -> None:
    if len(data) < 4:
        return
    pkt_len = struct.unpack_from(">H", data, 2)[0]
    if pkt_len < 12:
        return

    pkt_id = struct.unpack_from(">H", data, 6)[0]
    m_hdr  = data[8:12]

    def start_new_frame(payload: bytes) -> None:
        if session.cur_image and not session.frame_is_bad:
            if session.on_frame:
                session.on_frame(session)
        session.frame_is_bad = False
        session.cur_image    = [payload]
        session.rcv_seq_id   = pkt_id

    if m_hdr == FRAME_HEADER:
        if data[12] == 3:                           # JPEG stream_type
            to_read = pkt_len - 4 - 32
            if to_read > 0:
                start_new_frame(data[40:40 + to_read])
        # skip audio (stream_type 6)
    else:
        payload = data[8:8 + pkt_len - 4]
        if m_hdr == JPEG_HEADER:
            start_new_frame(payload)
        else:
            if pkt_id <= session.rcv_seq_id:
                return
            if pkt_id > session.rcv_seq_id + 1:
                session.frame_is_bad = True
                return
            session.rcv_seq_id = pkt_id
            session.cur_image.append(payload)

# ─── Incoming packet router ───────────────────────────────────────────────────

def handle_control_cmd(session: Session, data: bytes) -> None:
    cmd_id      = struct.unpack_from(">H", data, 10)[0]
    payload_len = struct.unpack_from("<H", data, 12)[0]   # LE
    if len(data) > 20 and payload_len > len(data):
        payload_len = len(data) - 20
    if payload_len > 4:
        buf = bytearray(data[20:20 + payload_len - 4])
        xq_dec(buf, len(buf), 4)
        data = data[:20] + bytes(buf) + data[20 + payload_len - 4:]

    if cmd_id == CtrlCmd.ConnectUserAck:
        session.ticket = data[24:28]
        print(f"[{session.dev.dev_id}] Logged in, starting video...")
        session.send(build_video_resolution(session, 2))
        session.send(build_start_video(session))

def handle_drw(session: Session, data: bytes) -> None:
    session.send(make_drw_ack(data))
    m_stream = data[5]
    if   m_stream == 1: deal_with_data(session, data)
    elif m_stream == 0: handle_control_cmd(session, data)

def handle_incoming(session: Session, data: bytes) -> None:
    session.last_rx = time.monotonic()
    cmd = struct.unpack_from(">H", data)[0]

    if   cmd == Cmd.P2pRdy:      session.send(build_usr_chk(session))
    elif cmd == Cmd.P2PAlive:    session.send(make_p2p_alive())
    elif cmd == Cmd.P2PAliveAck: pass
    elif cmd == Cmd.Drw:         handle_drw(session, data)
    elif cmd == Cmd.DrwAck:
        count = struct.unpack_from(">H", data, 6)[0]
        for i in range(count):
            pkt_id = struct.unpack_from(">H", data, 8 + i * 2)[0]
            session.unacked.pop(pkt_id, None)
    elif cmd == Cmd.Close:
        session.send(make_drw_ack(data))
        session._disconnect()

# ─── asyncio UDP protocol ─────────────────────────────────────────────────────

class SessionProtocol(asyncio.DatagramProtocol):
    def __init__(self, session: Session) -> None:
        self.session = session

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.session.transport = transport
        self.session.send(make_p2p_rdy(self.session.dev))
        self.session.started = True

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        handle_incoming(self.session, data)

    def error_received(self, exc: Exception) -> None:
        print(f"Session error: {exc}")

    def connection_lost(self, exc: Exception | None) -> None:
        pass

class DiscoveryProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_discover: callable) -> None:
        self.on_discover   = on_discover
        self.transport     = None
        self._lan_search   = make_lan_search()

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport
        sock = transport.get_extra_info("socket")
        sock.setsockopt(__import__("socket").SOL_SOCKET,
                        __import__("socket").SO_BROADCAST, 1)

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        cmd = struct.unpack_from(">H", data)[0]
        if cmd == Cmd.PunchPkt:
            self.on_discover(parse_punch_pkt(data), addr)

    def error_received(self, exc: Exception) -> None:
        print(f"Discovery error: {exc}")

    def send_lan_search(self, ip: str) -> None:
        if self.transport:
            self.transport.sendto(self._lan_search, (ip, 32108))

# ─── Keepalive loop ───────────────────────────────────────────────────────────

async def keepalive_loop(session: Session) -> None:
    while session.started and session.transport:
        await asyncio.sleep(0.4)
        delta = time.monotonic() - session.last_rx
        if not session.started:
            break
        if delta > 0.6:
            session.send(make_p2p_alive())
        if delta > 5.0:
            session._disconnect()
            break

# ─── Discovery loop ───────────────────────────────────────────────────────────

async def discovery_loop(proto: DiscoveryProtocol, ip: str) -> None:
    while True:
        proto.send_lan_search(ip)
        await asyncio.sleep(3)

# ─── OpenCV display ───────────────────────────────────────────────────────────

class FrameDisplay:
    """Thread-safe frame queue; OpenCV imshow runs in the asyncio thread via call_soon."""

    def __init__(self) -> None:
        self._latest: bytes | None = None
        self._window_open = False

    def push(self, jpeg: bytes) -> None:
        self._latest = jpeg

    def show_pending(self) -> bool:
        """Call from the main loop tick. Returns False when the window is closed."""
        if self._latest is None:
            return True
        jpeg = self._latest
        self._latest = None

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return True

        cv2.imshow("YsxLite Camera", frame)
        self._window_open = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:          # q or Esc → quit
            cv2.destroyAllWindows()
            return False
        return True

# ─── Main ─────────────────────────────────────────────────────────────────────

async def run(discovery_ip: str) -> None:
    loop    = asyncio.get_running_loop()
    display = FrameDisplay()
    sessions: dict[str, Session] = {}

    def on_frame(session: Session) -> None:
        jpeg = b"".join(session.cur_image)
        display.push(jpeg)

    def on_disconnect(session: Session) -> None:
        sessions.pop(session.dev.dev_id, None)

    def on_discover(dev: Device, addr: tuple) -> None:
        if dev.dev_id in sessions:
            return
        print(f"Discovered camera {dev.dev_id} at {addr[0]}")

        session = Session(
            dev=dev, dst_ip=addr[0], dst_port=addr[1],
            on_frame=on_frame, on_disconnect=on_disconnect,
        )
        sessions[dev.dev_id] = session

        async def _connect():
            await loop.create_datagram_endpoint(
                lambda: SessionProtocol(session),
                family=__import__("socket").AF_INET,
            )
            asyncio.create_task(keepalive_loop(session))

        asyncio.create_task(_connect())

    # Start discovery socket
    disc_transport, disc_proto = await loop.create_datagram_endpoint(
        lambda: DiscoveryProtocol(on_discover),
        local_addr=("0.0.0.0", 0),
        family=__import__("socket").AF_INET,
    )
    asyncio.create_task(discovery_loop(disc_proto, discovery_ip))
    print(f"Searching for cameras on {discovery_ip}...")

    # Main display tick
    try:
        while True:
            if not display.show_pending():
                break
            await asyncio.sleep(0.01)   # ~100 fps poll; OpenCV waitKey handles the rest
    except KeyboardInterrupt:
        pass
    finally:
        disc_transport.close()
        for s in list(sessions.values()):
            s.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    discovery_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.255"
    asyncio.run(run(discovery_ip))
