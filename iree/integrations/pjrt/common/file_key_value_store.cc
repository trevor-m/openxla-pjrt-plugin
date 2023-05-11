// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "file_key_value_store.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <thread>

namespace iree::pjrt {

FileKeyValueStore::FileKeyValueStore(const std::string& base_dir) {
  base_dir_ = base_dir;
  std::filesystem::create_directories(base_dir_);
}

std::string FileKeyValueStore::GetPath(const std::string& key) const {
  return base_dir_ + "/" + key;
}

bool FileKeyValueStore::Set(const std::string& key,
                            const std::vector<uint8_t>& value) {
  std::string path = GetPath(key);
  int fd = open(path.c_str(), O_RDWR | O_CREAT, 0666);
  int rc = flock(fd, LOCK_EX);
  if (rc) {
    // printf("flock failed, rc = %d\n", rc);
    return false;
  }
  ssize_t count = write(fd, value.data(), value.size());
  // printf("write() %ld bytes\n", count);
  close(fd);
  if (count != value.size()) {
    // printf("written %ld != %ld\n", count, value.size());
    return false;
  }
  return true;
}

bool FileKeyValueStore::Get(const std::string& key,
                            std::vector<uint8_t>& value) {
  std::string path = GetPath(key);

  int fd = -1;
  while (1) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
      if (errno == ENOENT) {
        // If the file does not exist yet, wait for it.
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      } else {
        // printf("open() failed, errno = %d\n", errno);
      }
    } else {
      int rc = flock(fd, LOCK_EX);
      if (rc) {
        printf("flock failed, rc = %d\n", rc);
        return false;
      }
      size_t size = lseek(fd, 0, SEEK_END);
      lseek(fd, 0, SEEK_SET);
      // printf("file size = %ld\n", size);
      if (size == 0) {
        // It is open, but no data yet. Keep waiting.
        close(fd);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        continue;
      } else {
        value.resize(size);
        ssize_t count = read(fd, value.data(), size);
        close(fd);
        if (count != size) {
          // printf("read failed %ld != %ld\n", count, size);
          return false;
        }
        return true;
      }
    }
  }

  return true;
}

}  // namespace iree::pjrt
