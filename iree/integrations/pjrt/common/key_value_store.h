// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_KEY_VALUE_STORE_H_
#define IREE_PJRT_PLUGIN_PJRT_KEY_VALUE_STORE_H_

#include <string>
#include <vector>

namespace iree::pjrt {

class KeyValueStore {
 public:
  virtual ~KeyValueStore();
  virtual bool Set(const std::string& key,
                   const std::vector<uint8_t>& value) = 0;
  virtual bool Get(const std::string& key, std::vector<uint8_t>& value) = 0;
};

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_KEY_VALUE_STORE_H_
