#!/usr/bin/env python3

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class DummyRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write('<html><body><h1>hi!</h1></body></html>'.encode())


def main(argv):
    if len(argv) != 2:
        print(f'Usage: {argv[0]} <PORT>', file=sys.stderr)
        return 1

    port = int(argv[1])
    srv = HTTPServer(('localhost', port), DummyRequestHandler)
    srv.serve_forever()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
