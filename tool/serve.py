"""
Simple HTTP server to serve the frontend application.
Run this and open http://localhost:8000 in your browser.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).parent / "frontend"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"✨ Unveiling Digital Mirrors - Pose Analysis Tool")
        print(f"=" * 50)
        print(f"Server running at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop")
        print(f"=" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
