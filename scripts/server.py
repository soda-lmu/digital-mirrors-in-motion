"""
Local HTTP server for Unveiling Digital Mirrors.
Serves frontend/ (HTML, JS, CSS, data/) on http://localhost:8000.

Usage:
    python scripts/server.py
Then open http://localhost:8000 in browser.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
ROOT_DIRECTORY = Path(__file__).parent.parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIRECTORY), **kwargs)

    def do_GET(self):
        if self.path.startswith('/api/list'):
            import json, urllib.parse
            media_type = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get('type', ['videos'])[0]
            target = ROOT_DIRECTORY / 'data' / media_type
            files = sorted([f.name for f in target.iterdir() if f.is_file() and f.suffix.lower() in {'.mp4', '.webm', '.mov', '.jpg', '.jpeg', '.png', '.webp'}]) if target.exists() else []
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(files).encode())
            return
        return super().do_GET()

if __name__ == "__main__":
    os.chdir(ROOT_DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Unveiling Digital Mirrors - Pose Analysis Tool")
        print(f"=" * 50)
        print(f"Server running at: http://localhost:{PORT}/frontend/index.html")
        print(f"Press Ctrl+C to stop")
        print(f"=" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
