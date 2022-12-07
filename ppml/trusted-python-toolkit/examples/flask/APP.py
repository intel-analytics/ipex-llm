from flask import Flask, request
app = Flask(__name__)

@app.route('/<name>', methods = ['GET', 'POST'])
def hello_world(name):
    if request.method == 'GET':
        return 'Hello ' + name + ' GET'
    else:
        return 'Hello ' + name + ' POST'

if __name__ == '__main__':
   app.run(host='0.0.0.0')

