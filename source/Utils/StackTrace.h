#ifdef CUP2D_BACKWARD_CPP
void enableStackTraceSignalHandling();
#else
inline void enableStackTraceSignalHandling() { }
#endif
