/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 */

#define _GNU_SOURCE

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include "ias.h"
#include "sgx_attest.h"
#include "util.h"

/** Default base URL for IAS API endpoints. Remove "/dev" for production environment. */
#define IAS_URL_BASE "https://api.trustedservices.intel.com/sgx/dev"

/** Default URL for IAS "verify attestation evidence" API endpoint. */
#define IAS_URL_REPORT IAS_URL_BASE "/attestation/v4/report"

/** Default URL for IAS "Retrieve SigRL" API endpoint. EPID group id is added at the end. */
#define IAS_URL_SIGRL IAS_URL_BASE "/attestation/v4/sigrl"

struct option g_options[] = {
    { "help", no_argument, 0, 'h' },
    { "verbose", no_argument, 0, 'v' },
    { "msb", no_argument, 0, 'm' },
    { "quote-path", required_argument, 0, 'q' },
    { "api-key", required_argument, 0, 'k' },
    { "nonce", required_argument, 0, 'n' },
    { "report-path", required_argument, 0, 'r' },
    { "sig-path", required_argument, 0, 's' },
    { "cert-path", required_argument, 0, 'c' },
    { "gid", required_argument, 0, 'g' },
    { "sigrl-path", required_argument, 0, 'i' },
    { "report-url", required_argument, 0, 'R' },
    { "sigrl-url", required_argument, 0, 'S' },
    { 0, 0, 0, 0 }
};

static void usage(const char* exec) {
    INFO("Usage: %s <request> [options]\n", exec);
    INFO("Available requests:\n");
    INFO("  sigrl                     Retrieve signature revocation list for a given EPID group\n");
    INFO("  report                    Verify attestation evidence (quote)\n");
    INFO("Available general options:\n");
    INFO("  --help, -h                Display this help\n");
    INFO("  --verbose, -v             Enable verbose output\n");
    INFO("  --msb, -m                 Print/parse hex strings in big-endian order\n");
    INFO("  --api-key, -k STRING      IAS API key\n");
    INFO("Available sigrl options:\n");
    INFO("  --gid, -g STRING          EPID group ID (hex string)\n");
    INFO("  --sigrl-path, -i PATH     Path to save SigRL to\n");
    INFO("  --sigrl-url, -S URL       URL for the IAS SigRL endpoint (default:\n"
         "                            %s)\n", IAS_URL_SIGRL);
    INFO("Available report options:\n");
    INFO("  --quote-path, -q PATH     Path to quote to submit\n");
    INFO("  --report-path, -r PATH    Path to save IAS report to\n");
    INFO("  --sig-path, -s PATH       Path to save IAS report's signature to\n");
    INFO("  --nonce, -n STRING        Nonce to use (optional)\n");
    INFO("  --cert-path, -c PATH      Path to save IAS certificate to (optional)\n");
    INFO("  --report-url, -R URL      URL for the IAS attestation report endpoint (default:\n"
         "                            %s)\n", IAS_URL_REPORT);
}

static int report(struct ias_context_t* ias, const char* quote_path, const char* nonce,
                  const char* report_path, const char* sig_path, const char* cert_path) {
    int ret = -1;
    void* quote_data = NULL;

    if (!quote_path) {
        ERROR("Quote path not specified\n");
        goto out;
    }

    size_t quote_size = 0;
    quote_data = read_file(quote_path, &quote_size, /*buffer=*/NULL);
    if (!quote_data) {
        ERROR("Failed to read quote file '%s'\n", quote_path);
        goto out;
    }

    if ((size_t)quote_size < sizeof(sgx_quote_t)) {
        ERROR("Quote is too small\n");
        goto out;
    }

    sgx_quote_t* quote = (sgx_quote_t*)quote_data;
    if ((size_t)quote_size < sizeof(sgx_quote_t) + quote->signature_size) {
        ERROR("Quote is too small\n");
        goto out;
    }
    quote_size = sizeof(sgx_quote_t) + quote->signature_size;

    ret = ias_verify_quote(ias, quote_data, quote_size, nonce, report_path, sig_path, cert_path);
    if (ret != 0) {
        ERROR("Failed to submit quote to IAS\n");
        goto out;
    }

    INFO("IAS submission successful\n");
out:
    free(quote_data);
    return ret;
}

static int sigrl(struct ias_context_t* ias, const char* gid_str, const char* sigrl_path) {
    uint8_t gid[4];

    if (parse_hex(gid_str, gid, sizeof(gid), NULL) != 0) {
        ERROR("Invalid EPID group ID\n");
        return -1;
    }

    size_t sigrl_size = 0;
    void* sigrl = NULL;
    int ret = ias_get_sigrl(ias, gid, &sigrl_size, &sigrl);
    if (ret == 0) {
        if (sigrl_size == 0) {
            INFO("No SigRL for given EPID group ID %s\n", gid_str);
        } else {
            DBG("SigRL size: %zu\n", sigrl_size);
            ret = write_file(sigrl_path, sigrl_size, sigrl);
        }
    }
    free(sigrl);
    return ret;
}

int main(int argc, char* argv[]) {
    int option              = 0;
    char* mode              = NULL;
    char* quote_path        = NULL;
    char* api_key           = NULL;
    char* nonce             = NULL;
    char* report_path       = NULL;
    char* sig_path          = NULL;
    char* cert_path         = NULL;
    char* gid               = NULL;
    char* sigrl_path        = NULL;
    const char* report_url  = IAS_URL_REPORT;
    const char* sigrl_url   = IAS_URL_SIGRL;
    endianness_t endian     = ENDIAN_LSB;

    // parse command line
    while (true) {
        option = getopt_long(argc, argv, "hvmq:k:n:r:s:c:a:g:i:R:S:", g_options, NULL);
        if (option == -1)
            break;

        switch (option) {
            case 'h':
                usage(argv[0]);
                return 0;
            case 'v':
                set_verbose(true);
                break;
            case 'm':
                endian = ENDIAN_MSB;
                break;
            case 'q':
                quote_path = optarg;
                break;
            case 'k':
                api_key = optarg;
                break;
            case 'n':
                nonce = optarg;
                break;
            case 'r':
                report_path = optarg;
                break;
            case 's':
                sig_path = optarg;
                break;
            case 'c':
                cert_path = optarg;
                break;
            case 'g':
                gid = optarg;
                break;
            case 'i':
                sigrl_path = optarg;
                break;
            case 'R':
                report_url = optarg;
                break;
            case 'S':
                sigrl_url = optarg;
                break;
            default:
                usage(argv[0]);
                return -1;
        }
    }

    set_endianness(endian);

    if (optind >= argc) {
        ERROR("Request not specified\n");
        usage(argv[0]);
        return -1;
    }

    mode = argv[optind++];

    if (!api_key) {
        ERROR("API key not specified\n");
        return -1;
    }

    struct ias_context_t* ias = ias_init(api_key, report_url, sigrl_url);
    if (!ias) {
        ERROR("Failed to initialize IAS library\n");
        return -1;
    }

    if (mode[0] == 'r') {
        if (!report_path || !sig_path) {
            ERROR("Report or signature path not specified\n");
            return -1;
        }
        return report(ias, quote_path, nonce, report_path, sig_path, cert_path);
    } else if (mode[0] == 's') {
        if (!sigrl_path) {
            ERROR("SigRL path not specified\n");
            return -1;
        }
        return sigrl(ias, gid, sigrl_path);
    }

    ERROR("Invalid request\n");
    usage(argv[0]);
    return -1;
}
