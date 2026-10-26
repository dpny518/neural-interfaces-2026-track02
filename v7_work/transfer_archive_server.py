"""Temporary read-only, single-artifact server bound to the user's tailnet IP."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading

ARCHIVE = Path('/mnt/18t/neural-interfaces-2026-track02-v7/v7_work/deliverables/reproduce.zip')
SHA = '7a101ef47340cbb4d2b9a331329cd35b61ca1fb8476c5aef8aece8632cb980ff'


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/' + SHA:
            self.send_error(404)
            return
        size = ARCHIVE.stat().st_size
        match = re.fullmatch(r'bytes=(\d+)-(\d*)', self.headers.get('Range', ''))
        if not match:
            self.send_error(400, 'Explicit byte range required')
            return
        lo = int(match[1])
        hi = int(match[2]) if match[2] else size - 1
        if not 0 <= lo <= hi < size:
            self.send_error(416)
            return
        self.send_response(206)
        self.send_header('Content-Length', str(hi - lo + 1))
        self.send_header('Content-Range', f'bytes {lo}-{hi}/{size}')
        self.send_header('X-Archive-SHA256', SHA)
        self.end_headers()
        try:
            with ARCHIVE.open('rb') as handle:
                handle.seek(lo)
                remaining = hi - lo + 1
                while remaining:
                    block = handle.read(min(262144, remaining))
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    assert ARCHIVE.stat().st_size == 1269782187
    ThreadingHTTPServer.request_queue_size = 128
    server = ThreadingHTTPServer(('100.71.130.36', 18767), Handler)
    timer = threading.Timer(3600, server.shutdown)
    timer.daemon = True
    timer.start()
    print('TAILNET_SINGLE_ARCHIVE_SERVER_READY', flush=True)
    server.serve_forever()
