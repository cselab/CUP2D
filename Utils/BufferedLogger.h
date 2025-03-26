#ifndef CubismUP_3D_utils_BufferedLogger_h
#define CubismUP_3D_utils_BufferedLogger_h
#include <sstream>
#include <unordered_map>
class BufferedLogger {
  struct Stream {
    std::stringstream stream;
    int requests_since_last_flush = 0;
    Stream(const Stream &c) {}
    Stream() {}
  };
  typedef std::unordered_map<std::string, Stream> container_type;
  container_type files;
  void flush(container_type::iterator it);

public:
  ~BufferedLogger() { flush(); }
  std::stringstream &get_stream(const std::string &filename);
  inline void flush(void) {
    for (auto it = files.begin(); it != files.end(); ++it)
      flush(it);
  }
};
extern BufferedLogger logger;
#endif
