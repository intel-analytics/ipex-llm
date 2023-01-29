import requests

flask_address = your_flask_address
url = flask_address + '/World!'
res = requests.post(url=url)
print(res.text)
