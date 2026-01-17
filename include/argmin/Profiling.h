#pragma once

// Profiling support for ArgMin (optional)
// This is a lightweight profiling macro that can be enabled with ARGMIN_ENABLE_PROFILING

#ifdef ARGMIN_ENABLE_PROFILING
  #include <chrono>
  #include <string>

  namespace ArgMin {
  /**
   * RAII-based profiler.
   * Construct this in a scope to profile execution time.
   * Time is logged in milliseconds when the object is destroyed.
   */
  class Profiler {
   public:
    Profiler(std::string functionName) : functionName(functionName) {
      startTime = std::chrono::system_clock::now();
    }

    ~Profiler() {
      std::chrono::duration<double, std::milli> duration =
          std::chrono::system_clock::now() - startTime;
      // Could log here if logging is enabled, but for now just measure
      (void)duration; // Suppress unused warning
    }

   private:
    std::string functionName;
    std::chrono::time_point<std::chrono::system_clock> startTime;
  };
  }  // namespace ArgMin

  #define PROFILE(sectionName) ArgMin::Profiler __argmin_profiler_(sectionName);
#else
  // No-op when profiling is disabled (default)
  #define PROFILE(sectionName) (void)0
#endif
