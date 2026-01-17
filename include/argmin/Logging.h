/*
 * This file is used to define the log level globally.
 * It can also be used to define new helper macros.
 *
 * This is a wrapper to make spdlog optional for ArgMin.
 */

#pragma once

#ifdef ARGMIN_USE_SPDLOG
  #define SPDLOG_HEADER_ONLY
  #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
  #include <spdlog/fmt/ostr.h>
  #include <spdlog/spdlog.h>

  /// Define trace macro to force a log level set.
  #if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
  #define LOG_TRACE(...)                       \
    {                                          \
      spdlog::set_level(spdlog::level::trace); \
      SPDLOG_TRACE(__VA_ARGS__);               \
    }
  #else
  #define LOG_TRACE(...) (void)0
  #endif

  // Info
  #if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
  #define LOG_INFO(...) \
    { SPDLOG_INFO(__VA_ARGS__); }
  #else
  #define LOG_INFO(...) (void)0
  #endif

  // Warn
  #if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
  #define LOG_WARN(...) \
    { SPDLOG_WARN(__VA_ARGS__); }
  #else
  #define LOG_WARN(...) (void)0
  #endif

  // Error
  #if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
  #define LOG_ERROR(...) \
    { SPDLOG_ERROR(__VA_ARGS__); }
  #else
  #define LOG_ERROR(...) (void)0
  #endif

#else
  // No-op macros when spdlog is disabled
  #define LOG_TRACE(...) (void)0
  #define LOG_INFO(...) (void)0
  #define LOG_WARN(...) (void)0
  #define LOG_ERROR(...) (void)0
#endif

// CHECK macro for runtime assertions (always enabled)
#include <cstdlib>
#define CHECK(condition, ...) \
  {                           \
    if (!(condition)) {       \
      LOG_ERROR(__VA_ARGS__); \
      abort();                \
    }                         \
  }
