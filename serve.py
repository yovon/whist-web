"""serve.py — tiny static file server for the Pyodide proof-of-concept.

Pyodide must fetch the bundle over http:// (file:// is blocked by the browser),
so open the page through this server, not by double-clicking index.html.

Run:
    python serve.py
then open http://localhost:8000/ in a browser.
"""
import http.server
import os
import socketserver

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = 8000

handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), handler) as httpd:
    print(f"Serving {os.getcwd()}")
    print(f"Open http://localhost:{PORT}/  (Ctrl-C to stop)")
    httpd.serve_forever()
