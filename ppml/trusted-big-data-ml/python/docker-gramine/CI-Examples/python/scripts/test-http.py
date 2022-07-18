#!/usr/bin/env python3

import sys
import urllib.request

request = urllib.request.Request("http://" + sys.argv[1] + ":" + sys.argv[2] + "/index.html")
opener = urllib.request.build_opener()
response = opener.open(request, timeout=10)
while True:
    data = response.read(1024)
    if data:
        print(data.decode())
    else:
        break
