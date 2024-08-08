from multiprocessing import Process
from multiprocessing.connection import Listener
import time
import daemon

def browse_server_loop(address: tuple[str, int]) -> None:
    """Runs the server loop"""
    with Listener(address) as listener:
        print("Starting browse server at", address)
        while True:
            with listener.accept() as conn:
                print("connection accepted from", listener.last_accepted)
                request = conn.recv()
                print("received request:", request)
                conn.send(request)


def start_or_get_server_url() -> str:
    address = ("localhost", 6000)  # family is deduced to be 'AF_INET'

    """Starts the server and returns the URL"""
    daemon.create_daemon(target=browse_server_loop, args=(address,))
    time.sleep(0.1)
    return address
