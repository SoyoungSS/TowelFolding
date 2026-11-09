"""Transport and lockstep state-machine tests; run with towel-newton Python."""
import importlib.util
from pathlib import Path
import socket
import struct
import sys
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"simulation"))
from bridge_transport import send_message, receive_message, MAX_MESSAGE
spec = importlib.util.spec_from_file_location("worker",ROOT/"scripts/isaac/live_newton_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FakeEnv:
    def __init__(self):
        self.episode = -1
        self.steps = 0
        self.scene = SimpleNamespace(fold=SimpleNamespace(metrics=None,history=["settle"]))
    def reset(self,seed):
        self.episode += 1
        self.frame = 0
        return self.get_state()
    def start_expert(self):
        pass
    def get_state(self):
        return {"episode_id":self.episode,"frame_id":self.frame,"phase":"settle"}
    def metadata(self):
        return {"test":True}
    def expert_action(self):
        return None
    def step(self,action):
        self.frame += 1
        self.steps += 1
        return self.get_state()


class BridgeTests(unittest.TestCase):
    def test_fragmented_message_and_disconnect(self):
        a,b = socket.socketpair()
        with a,b:
            b.settimeout(2)
            packet = struct.pack("!I",7)+b'{"x":1}'
            def write_fragments():
                for byte in packet:
                    a.sendall(bytes([byte]))
                a.shutdown(socket.SHUT_WR)
            thread = threading.Thread(target=write_fragments)
            thread.start()
            self.assertEqual(receive_message(b),{"x":1})
            with self.assertRaises(EOFError):
                receive_message(b)
            thread.join(2)

    def test_oversized_packet_rejected_before_payload(self):
        a,b = socket.socketpair()
        with a,b:
            a.sendall(struct.pack("!I",MAX_MESSAGE+1))
            with self.assertRaises(ValueError):
                receive_message(b)

    def worker_pair(self):
        a,b = socket.socketpair()
        b.settimeout(2)
        env,errors = FakeEnv(),[]
        def run():
            try:
                worker.serve(a,env,0)
            except BaseException as error:
                errors.append(error)
            finally:
                a.close()
        thread=threading.Thread(target=run,daemon=True)
        thread.start()
        return b,env,errors,thread

    def test_acknowledged_advance_reset_quit(self):
        client,env,errors,thread=self.worker_pair()
        with client:
            self.assertEqual(receive_message(client)["state"]["frame_id"],0)
            self.assertEqual(env.steps,0)  # no free-running physics
            send_message(client,{"command":"advance","request_id":1,"episode_id":0,"frame_id":0,"steps":3})
            self.assertEqual(receive_message(client)["state"]["frame_id"],3)
            send_message(client,{"command":"reset","request_id":2,"episode_id":0,"frame_id":3})
            self.assertEqual(receive_message(client)["state"],{"episode_id":1,"frame_id":0,"phase":"settle"})
            send_message(client,{"command":"quit","request_id":3})
            self.assertEqual(receive_message(client)["type"],"bye")
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors,[])

    def test_wrong_ack_does_not_advance_physics(self):
        client,env,errors,thread=self.worker_pair()
        with client:
            receive_message(client)
            send_message(client,{"command":"advance","request_id":1,"episode_id":0,"frame_id":99})
            with self.assertRaises(EOFError):
                receive_message(client)
        thread.join(2)
        self.assertEqual(env.steps,0)
        self.assertIsInstance(errors[0],ValueError)


if __name__ == "__main__":
    unittest.main()
