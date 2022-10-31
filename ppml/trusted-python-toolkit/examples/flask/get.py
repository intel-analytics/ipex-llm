import requests

flask_address = your_flask_address
url = flask_address + '/World!'
res = requests.get(url=url)
print(res.text)

