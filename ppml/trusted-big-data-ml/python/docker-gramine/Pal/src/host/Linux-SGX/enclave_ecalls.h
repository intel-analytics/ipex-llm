/* SPDX-License-Identifier: LGPL-3.0-or-later */

#ifndef _ENCLAVE_ECALLS_H_
#define _ENCLAVE_ECALLS_H_

void handle_ecall(long ecall_index, void* ecall_args, void* exit_target, void* enclave_base_addr);

#endif /* _ENCLAVE_ECALLS_H_ */
