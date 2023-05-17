// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_CHANNEL_PROVIDER_H_
#define IREE_PJRT_PLUGIN_PJRT_CHANNEL_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "key_value_store.h"

namespace iree::pjrt {
// Creates a key/value-based collective channel provider.
// Users should create one provider and share it across all
// devices used within the process.
iree_status_t channel_provider_create(
    iree_allocator_t host_allocator, KeyValueStore* kvs, int32_t default_rank,
    int32_t default_count, iree_hal_channel_provider_t** out_channel_provider);

}  // namespace iree::pjrt
#endif  // IREE_PJRT_PLUGIN_PJRT_CHANNEL_PROVIDER_H_
