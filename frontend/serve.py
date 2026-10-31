"""Local frontend preview for the existing pricing API on 127.0.0.1:8000.

Run from the project root: python frontend/serve.py
The browser uses this same-origin server for static files and a small API proxy.
"""

import csv
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


FRONTEND_DIR = Path(__file__).resolve().parent
SELECTED_CSV = FRONTEND_DIR.parent / "models_2y" / "selected_products_online_retail.csv"
BACKEND = "http://127.0.0.1:8000"
GET_ROUTES = {"/health", "/model_info", "/products"}
POST_ROUTES = {"/recommend", "/recommend/batch"}


class FrontendHandler(BaseHTTPRequestHandler):
    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json_error(self, status, detail):
        self.send_bytes(status, json.dumps({"detail": detail}).encode("utf-8"),
                        "application/json; charset=utf-8")

    def proxy(self, backend_path, query, body=None):
        url = BACKEND + backend_path + ("?" + query if query else "")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = Request(url, data=body, headers=headers, method=self.command)
        try:
            with urlopen(request, timeout=15) as response:
                self.send_bytes(response.status, response.read(),
                                response.headers.get("Content-Type", "application/json"))
        except HTTPError as error:
            self.send_bytes(error.code, error.read(),
                            error.headers.get("Content-Type", "application/json"))
        except URLError:
            self.send_json_error(502, "API gốc tại 127.0.0.1:8000 chưa chạy.")

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        static = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        if path in static:
            name, content_type = static[path]
            self.send_bytes(200, (FRONTEND_DIR / name).read_bytes(), content_type)
        elif path == "/selected-products.json":
            try:
                with SELECTED_CSV.open(newline="", encoding="utf-8-sig") as source:
                    rows = list(csv.DictReader(source))
                self.send_bytes(200, json.dumps(rows, ensure_ascii=False).encode("utf-8"),
                                "application/json; charset=utf-8")
            except OSError:
                self.send_json_error(500, "Không tìm thấy kết quả chọn 20 sản phẩm ở models_2y.")
        elif path == "/docs":
            self.send_response(302)
            self.send_header("Location", BACKEND + "/docs")
            self.end_headers()
        elif path.startswith("/api/"):
            backend_path = path[4:]
            allowed = backend_path in GET_ROUTES or (
                backend_path.startswith("/demand_curve/") and
                re.fullmatch(r"[A-Za-z0-9]+", backend_path.removeprefix("/demand_curve/"))
            )
            if allowed:
                self.proxy(backend_path, parsed.query)
            else:
                self.send_json_error(404, "Endpoint không thuộc API hiện tại.")
        elif path == "/favicon.ico":
            self.send_bytes(204, b"", "image/x-icon")
        else:
            self.send_json_error(404, "Không tìm thấy tài nguyên.")

    def do_POST(self):
        parsed = urlsplit(self.path)
        backend_path = parsed.path[4:] if parsed.path.startswith("/api/") else ""
        if backend_path not in POST_ROUTES:
            self.send_json_error(404, "Endpoint không thuộc API hiện tại.")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json_error(400, "Content-Length không hợp lệ.")
            return
        if not 0 < length <= 1_000_000:
            self.send_json_error(413, "Kích thước yêu cầu không hợp lệ.")
            return
        self.proxy(backend_path, parsed.query, self.rfile.read(length))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), FrontendHandler)
    print("Frontend: http://127.0.0.1:8765/", flush=True)
    print("API gốc cần chạy tại http://127.0.0.1:8000/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
