"""Length-prefixed JSON on a local Unix socket; no pickle or network listener."""
import json
import struct

MAX_MESSAGE = 4 * 1024 * 1024


def send_message(sock, message):
    def convert(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(type(value).__name__)
    payload = json.dumps(message, default=convert, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError("Bridge message too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(sock):
    def exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = sock.recv(size-len(chunks))
            if not chunk:
                raise EOFError("Bridge peer disconnected")
            chunks.extend(chunk)
        return bytes(chunks)
    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid bridge message size")
    return json.loads(exact(size))
