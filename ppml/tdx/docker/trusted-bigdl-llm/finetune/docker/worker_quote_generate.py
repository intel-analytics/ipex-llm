import quote_generator
import argparse
import ssl, os
import base64
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--user_report_data", type=str, default="ppml")

args = parser.parse_args()

host = os.environ.get('HYDRA_BSTRAP_LOCALHOST').split('.')[0]
user_report_data = args.user_report_data
try:
    quote_b = quote_generator.generate_tdx_quote(user_report_data)
    quote = base64.b64encode(quote_b).decode('utf-8')
except Exception as e:
    quote = "quote generation failed: %s" % (e)

print("%s: %s"%(host, quote))