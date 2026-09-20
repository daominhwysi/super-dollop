import os
import sys
import http.server
import socketserver
from pathlib import Path

PORT = 2303
SERVE_DIR = Path(__file__).resolve().parent.parent / "backend" / "logs"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

def run():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Artifact Report Server running at http://localhost:{PORT}/artifact/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down server...")

if __name__ == "__main__":
    run()
