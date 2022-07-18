/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs for miscellaneous use.
 */

#include "api.h"
#include "assert.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

int _DkSystemTimeQuery(uint64_t* out_usec) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkRandomBitsRead(void* buffer, size_t size) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkSegmentBaseGet(enum pal_segment_reg reg, uintptr_t* addr) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkSegmentBaseSet(enum pal_segment_reg reg, uintptr_t addr) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkCpuIdRetrieve(uint32_t leaf, uint32_t subleaf, uint32_t values[4]) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkAttestationReport(const void* user_report_data, PAL_NUM* user_report_data_size,
                         void* target_info, PAL_NUM* target_info_size, void* report,
                         PAL_NUM* report_size) {
    __UNUSED(user_report_data);
    __UNUSED(user_report_data_size);
    __UNUSED(target_info);
    __UNUSED(target_info_size);
    __UNUSED(report);
    __UNUSED(report_size);
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkAttestationQuote(const void* user_report_data, PAL_NUM user_report_data_size, void* quote,
                        PAL_NUM* quote_size) {
    __UNUSED(user_report_data);
    __UNUSED(user_report_data_size);
    __UNUSED(quote);
    __UNUSED(quote_size);
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkGetSpecialKey(const char* name, void* key, size_t* key_size) {
    __UNUSED(name);
    __UNUSED(key);
    __UNUSED(key_size);
    return -PAL_ERROR_NOTIMPLEMENTED;
}

double _DkGetBogomips(void) {
    /* this has to be implemented */
    return 0.0;
}
