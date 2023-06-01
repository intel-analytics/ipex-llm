import attestation_service, quote_generator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", type=str, help='the url for attestation service', required=True)
    parser.add_argument("-t", "--as_type", type=str, help='the type of attestation service', default='BigDL')
    parser.add_argument("-i", "--app_id", type=str, help='the app id for attestation service', default='')
    parser.add_argument('-k', '--api_key', type=str, help='the api key for attestation service', default='')
    parser.add_argument('-O', '--quote_type', type=str, help='quote type', default='TDX')
    parser.add_argument('-o', '--policy_id', type=str, help='policy id', default='')
    parser.add_argument('-p', '--user_report', type=str, help='user report', default='ppml')
    args = parser.parse_args()

    quote = quote_generator.generate_tdx_quote(args.user_report)
    attestation_result = attestation_service.bigdl_attestation_service(args.url, args.app_id, args.api_key, quote, args.policy_id)
    if attestation_result == 0:
        print("Attestation Success!")
    if attestation_result == 1:
        print("WARNING: Attestation pass but BIOS or the software is out of date.")
        print("Attestation Success!")
    if attestation_result == -1:
        print("Attestation Failed!")