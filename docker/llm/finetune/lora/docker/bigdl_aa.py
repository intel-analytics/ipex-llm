import quote_generator
from flask import Flask, request
from configparser import ConfigParser
import ssl, os
import base64
import requests

app = Flask(__name__)
use_secure_cert = False
# 生成自签名的SSL证书
context = ssl.SSLContext(ssl.PROTOCOL_TLS)
context.load_cert_chain(certfile='server.crt', keyfile='server.key')

@app.route('/gen_quote', methods=['POST'])
def gen_quote():
    data = request.get_json()
    user_report_data = data.get('user_report_data')

    quote_b = quote_generator.generate_tdx_quote(user_report_data)
    quote = base64.b64encode(quote_b.encode()).decode('utf-8')

    return {'quote': quote}

if __name__ == '__main__':
    # if not os.path.exists("/dev/tdx-guest"):
    #     print("BigDL-AA: TDX device 'tdx-guest' not found, service stopped.")
    #     exit(1)
    # if app.config['launchtime_attest'] == "true":
    #     ret = attest_for_entrypoint()
    #     if ret < 0 :
    #         print("BigDL-AA: Attestation failed, service stopped.")
    #         exit(1)
    #     else:
    #         print("BigDL-AA: Attestation success!")
    
    print("BigDL-AA: Agent Started.")
    app.run(host='0.0.0.0', port=9870, ssl_context=context)
