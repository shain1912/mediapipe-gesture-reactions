"""웹 버전 실행: web 폴더를 로컬 서버로 띄우고 브라우저를 연다.

실행: python serve_web.py
(웹캠은 보안상 http://localhost 또는 https 에서만 동작하므로 파일을 직접 열면 안 됨)
"""
import functools
import http.server
import webbrowser

from export_web_model import WEB_DIR, WEB_MODEL_PATH, export

PORT = 8000


def main():
    if not WEB_MODEL_PATH.exists():
        export()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"{url} 에서 실행 중 (종료: Ctrl+C)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
