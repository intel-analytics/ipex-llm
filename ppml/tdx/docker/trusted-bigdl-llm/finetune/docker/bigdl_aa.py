import quote_generator
from flask import Flask, request
from configparser import ConfigParser
import ssl, os
import base64
import requests
import subprocess
import shlex

app = Flask(__name__)

@app.route('/gen_quote', methods=['POST'])
def gen_quote():
    data = request.get_json()
    user_report_data = data.get('user_report_data')
    try:
        quote_b = quote_generator.generate_tdx_quote(user_report_data)
        quote = base64.b64encode(quote_b).decode('utf-8')
        return {'quote': quote}
    except Exception as e:
        return {'quote': "quote generation failed: %s" % (e)}

@app.route('/attest', methods=['POST'])
def get_cluster_quote_list():
    data = request.get_json()
    user_report_data = data.get('user_report_data')
    quote_list = []

    try:
        quote_b = quote_generator.generate_tdx_quote(user_report_data)
        quote = base64.b64encode(quote_b).decode("utf-8")
        quote_list.append(("launcher", quote))
    except Exception as e:
        quote_list.append("launcher", "quote generation failed: %s" % (e))

    command = "sudo -u mpiuser -E bash /ppml/get_worker_quote.sh %s" % (shlex.quote(user_report_data))
    output = subprocess.check_output(command, shell=True)

    with open("/ppml/output/quote.log", "r") as quote_file:
        for line in quote_file:
            line = line.strip()
            if line:
                parts = line.split(":") 
                if len(parts) == 2:
                    quote_list.append((parts[0].strip(), parts[1].strip())) 
    return {"quote_list": dict(quote_list)}

if __name__ == '__main__':
    print("BigDL-AA: Agent Started.")
    port = int(os.environ.get('ATTESTATION_API_SERVICE_PORT'))
    enable_tls = os.environ.get('ENABLE_TLS')
    if enable_tls == 'true':
        context = ssl.SSLContext(ssl.PROTOCOL_TLS)
        context.load_cert_chain(certfile='/ppml/keys/server.crt', keyfile='/ppml/keys/server.key')
        # https_key_store_token = os.environ.get('HTTPS_KEY_STORE_TOKEN')
        # context.load_cert_chain(certfile='/ppml/keys/server.crt', keyfile='/ppml/keys/server.key', password=https_key_store_token)
        app.run(host='0.0.0.0', port=port, ssl_context=context)
    else:
        app.run(host='0.0.0.0', port=port)
