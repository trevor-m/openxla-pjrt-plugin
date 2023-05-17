// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "channel_provider.h"

#include <stdlib.h>

#include "iree/base/tracing.h"

namespace iree::pjrt {
typedef struct channel_provider_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree::pjrt::KeyValueStore* kvs;
  int32_t default_rank;
  int32_t default_count;
} channel_provider_t;

extern const iree_hal_channel_provider_vtable_t channel_provider_vtable;

static channel_provider_t* channel_provider_cast(
    iree_hal_channel_provider_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &channel_provider_vtable);
  return (channel_provider_t*)base_value;
}

static void channel_provider_destroy(
    iree_hal_channel_provider_t* base_channel_provider) {
  IREE_ASSERT_ARGUMENT(base_channel_provider);
  channel_provider_t* channel_provider =
      channel_provider_cast(base_channel_provider);
  iree_allocator_t host_allocator = channel_provider->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(host_allocator, channel_provider);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t query_default_rank_and_count(
    iree_hal_channel_provider_t* base_channel_provider, int32_t* out_rank,
    int32_t* out_count) {
  IREE_ASSERT_ARGUMENT(base_channel_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  channel_provider_t* channel_provider =
      channel_provider_cast(base_channel_provider);
  *out_rank = channel_provider->default_rank;
  *out_count = channel_provider->default_count;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t exchange_default_id(
    iree_hal_channel_provider_t* base_channel_provider, iree_byte_span_t id) {
  IREE_ASSERT_ARGUMENT(base_channel_provider);
  channel_provider_t* channel_provider =
      channel_provider_cast(base_channel_provider);
  const std::string default_key("default");

  IREE_TRACE_ZONE_BEGIN(z0);
  if (channel_provider->default_rank == 0) {
    std::vector value(id.data, id.data + id.data_length);
    if (!channel_provider->kvs->Set(default_key, value)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INTERNAL, "kvs->Set() failed");
    }
  } else {
    std::vector<uint8_t> value;
    if (!channel_provider->kvs->Get(default_key, value)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INTERNAL, "kvs->Get() failed");
    }
    assert(value.size() == id.data_length);
    memcpy(id.data, value.data(), id.data_length);
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

const iree_hal_channel_provider_vtable_t channel_provider_vtable = {
    /*.destroy=*/channel_provider_destroy,
    /*.query_default_rank_and_count=*/query_default_rank_and_count,
    /*.exchange_default_id=*/exchange_default_id,
    ///*.exchange_id_for_group=*/exchange_id_for_group,
};

iree_status_t channel_provider_create(
    iree_allocator_t host_allocator, iree::pjrt::KeyValueStore* kvs,
    int32_t default_rank, int32_t default_count,
    iree_hal_channel_provider_t** out_channel_provider) {
  IREE_ASSERT_ARGUMENT(kvs);
  IREE_ASSERT_ARGUMENT(out_channel_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  channel_provider_t* channel_provider = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*channel_provider),
                                (void**)&channel_provider));
  iree_hal_resource_initialize(&channel_provider_vtable,
                               &channel_provider->resource);
  channel_provider->host_allocator = host_allocator;
  channel_provider->kvs = kvs;
  channel_provider->default_rank = default_rank;
  channel_provider->default_count = default_count;
  *out_channel_provider = (iree_hal_channel_provider_t*)channel_provider;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

}  // namespace iree::pjrt
