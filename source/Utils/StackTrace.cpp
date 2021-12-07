#ifdef CUP2D_BACKWARD_CPP
#include "backward.hpp"

void enableStackTraceSignalHandling() {
  // Enable signal handling, prints the stack trace when the app crashes.
  static backward::SignalHandling sh;
}

#endif  // CUP2D_BACKWARD_CPP
