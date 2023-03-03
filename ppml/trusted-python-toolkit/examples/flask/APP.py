from flask import Flask, request
from flask import jsonify
app = Flask(__name__)

@app.route('/<name>', methods = ['GET', 'POST'])
def hello_world(name):
    if request.method == 'GET':
        return jsonify({"GET":'Hello ' + name + ' GET'})
    else:
        return jsonify({"POST":'Hello ' + name + ' POST'})

if __name__ == '__main__':
   app.run(host='0.0.0.0')

