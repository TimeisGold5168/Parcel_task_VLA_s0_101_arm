// BSP control process (item 11): owns the Feetech bus, runs the B-spline
// control loop at a fixed rate. Python (bsp/cpp_control.py) plans and
// writes each new segment into shared memory; this process samples it.
//
// Same rules as run_bsp.py Executor.control():
//   tau = (now - t_start) * fps * speed      (demo frames)
//   s   = clip(start + tau - now0, t0, t1 - 1e-6), target = spline(s)
//   smoothstep fade from the old segment, step clamp to +-max_step deg.
// B-spline: scipy BSpline(knots = cummax(col 0), ctrl = rows[:-(k+1)],
// k, extrapolate=False), evaluated with de Boor.
//
// Bus (only with --port; without it the loop runs dry, no I/O):
// Goal_Position (42) sync write each tick, Present_Position (56) sync
// read every --read-every ticks, published as the arm state.
//
//   bsp_ctl --shm /dev/shm/bsp_ctl --hz 50 [--port /dev/ttyACM0
//           --calib my_follower_arm.json] [--rt]
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "layout.h"

namespace {

double now_s() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ------------------------------------------------------------ B-spline
struct Seg {
  int rows = 0, dim = 0;
  double start = 0, now0 = 0, t0 = 0, t1 = 0;
  double knots[kMaxRows];
  double ctrl[kMaxRows * kMaxDim];
};

// scipy BSpline with extrapolate=False: NaN outside [t[k], t[n]].
void spline_eval(const Seg& g, int k, double x, double* out) {
  const int n = g.rows - k - 1;  // number of control points
  const double* t = g.knots;
  if (k < 1 || k > kMaxDeg || n <= k || !(x >= t[k] && x <= t[n])) {
    for (int d = 0; d < g.dim; ++d) out[d] = NAN;
    return;
  }
  int i = k;
  while (i < n - 1 && !(x < t[i + 1])) ++i;
  while (i > k && t[i] == t[i + 1]) --i;
  double dv[kMaxDeg + 1];
  for (int d = 0; d < g.dim; ++d) {
    for (int j = 0; j <= k; ++j) dv[j] = g.ctrl[(j + i - k) * kMaxDim + d];
    for (int r = 1; r <= k; ++r) {
      for (int j = k; j >= r; --j) {
        const double lo = t[j + i - k], hi = t[j + 1 + i - r];
        const double a = hi > lo ? (x - lo) / (hi - lo) : 0.0;
        dv[j] = (1.0 - a) * dv[j - 1] + a * dv[j];
      }
    }
    out[d] = dv[k];
  }
}

// ------------------------------------------------------------ bus
struct Motor {
  int id;
  bool degrees;
  int lo, hi, drive;
};

std::vector<Motor> read_calib(const std::string& path) {
  // my_follower_arm.json, motors in the robot order (LeRobot SO-101)
  const char* names[] = {"shoulder_pan", "shoulder_lift", "elbow_flex",
                         "wrist_flex", "wrist_roll", "gripper"};
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string js = ss.str();
  std::vector<Motor> ms;
  for (int m = 0; m < 6; ++m) {
    const size_t p = js.find(std::string("\"") + names[m] + "\"");
    if (p == std::string::npos) {
      fprintf(stderr, "calibration: no %s\n", names[m]);
      exit(2);
    }
    const std::string blk = js.substr(p, js.find('}', p) - p);
    auto num = [&](const char* key) {
      std::smatch mt;
      std::regex re(std::string("\"") + key + "\"\\s*:\\s*(-?\\d+)");
      if (!std::regex_search(blk, mt, re)) {
        fprintf(stderr, "calibration: %s.%s\n", names[m], key);
        exit(2);
      }
      return std::stoi(mt[1]);
    };
    ms.push_back({num("id"), m < 5, num("range_min"), num("range_max"),
                  num("drive_mode")});
  }
  return ms;
}

class Bus {
 public:
  bool open_port(const char* port) {
    fd_ = open(port, O_RDWR | O_NOCTTY);
    if (fd_ < 0) return false;
    termios tio{};
    cfmakeraw(&tio);
    cfsetispeed(&tio, B1000000);
    cfsetospeed(&tio, B1000000);
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 0;
    tcsetattr(fd_, TCSANOW, &tio);
    tcflush(fd_, TCIOFLUSH);
    return true;
  }
  void close_port() {
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
  }

  // Goal_Position, LeRobot units (deg, gripper 0-100) -> raw ticks
  void write_goal(const std::vector<Motor>& ms, const double* q) {
    std::vector<uint8_t> p = {42, 2};
    for (size_t m = 0; m < ms.size(); ++m) {
      const int raw = encode(ms[m], q[m]);
      p.push_back(ms[m].id);
      p.push_back(raw & 0xFF);
      p.push_back((raw >> 8) & 0xFF);
    }
    send(0xFE, 0x83, p);
  }

  // Present_Position -> LeRobot units. false on a missing reply.
  bool read_present(const std::vector<Motor>& ms, double* q) {
    std::vector<uint8_t> p = {56, 2};
    for (const auto& m : ms) p.push_back(m.id);
    tcflush(fd_, TCIFLUSH);
    send(0xFE, 0x82, p);
    for (size_t m = 0; m < ms.size(); ++m) {
      uint8_t buf[8];  // FF FF id len err lo hi chk
      if (!recv_n(buf, 8, 0.004)) return false;
      if (buf[0] != 0xFF || buf[1] != 0xFF || buf[2] != ms[m].id) {
        return false;
      }
      uint8_t sum = 0;
      for (int i = 2; i < 7; ++i) sum += buf[i];
      if (static_cast<uint8_t>(~sum) != buf[7]) return false;
      int raw = buf[5] | (buf[6] << 8);
      if (raw & 0x8000) raw = -(raw & 0x7FFF);  // sign bit 15
      q[m] = decode(ms[m], raw);
    }
    return true;
  }

 private:
  static int encode(const Motor& m, double v) {
    int raw;
    if (m.degrees) {
      const double mid = (m.lo + m.hi) / 2.0;
      raw = static_cast<int>(v * 4095.0 / 360.0 + mid);
    } else {
      if (m.drive) v = 100.0 - v;
      v = std::clamp(v, 0.0, 100.0);
      raw = static_cast<int>(v / 100.0 * (m.hi - m.lo) + m.lo);
    }
    if (raw < 0) raw = 0x8000 | (-raw);
    return raw;
  }
  static double decode(const Motor& m, int raw) {
    if (m.degrees) {
      const double mid = (m.lo + m.hi) / 2.0;
      return (raw - mid) * 360.0 / 4095.0;
    }
    const int b = std::clamp(raw, m.lo, m.hi);
    const double v = (b - m.lo) * 100.0 / (m.hi - m.lo);
    return m.drive ? 100.0 - v : v;
  }
  void send(uint8_t id, uint8_t instr, const std::vector<uint8_t>& p) {
    std::vector<uint8_t> pk = {0xFF, 0xFF, id,
                               static_cast<uint8_t>(p.size() + 2), instr};
    pk.insert(pk.end(), p.begin(), p.end());
    uint8_t sum = 0;
    for (size_t i = 2; i < pk.size(); ++i) sum += pk[i];
    pk.push_back(~sum);
    if (write(fd_, pk.data(), pk.size()) < 0) perror("bus write");
  }
  bool recv_n(uint8_t* buf, int n, double timeout) {
    const double end = now_s() + timeout;
    int got = 0;
    while (got < n && now_s() < end) {
      const ssize_t r = read(fd_, buf + got, n - got);
      if (r > 0) got += r;
    }
    return got == n;
  }
  int fd_ = -1;
};

// ------------------------------------------------------------ shm
Shm* map_shm(const char* path) {
  const int fd = open(path, O_RDWR);
  if (fd < 0) {
    perror(path);
    exit(2);
  }
  void* p = mmap(nullptr, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED,
                 fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(2);
  }
  return static_cast<Shm*>(p);
}

// Copy the Python-written block under its seqlock (writer: odd = busy).
bool read_input(Shm* s, ShmIn* out) {
  for (int tries = 0; tries < 100; ++tries) {
    const double a = s->in.seq;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (static_cast<int64_t>(a) % 2) continue;
    std::memcpy(out, &s->in, sizeof(ShmIn));
    std::atomic_thread_fence(std::memory_order_acquire);
    if (s->in.seq == a) return true;
  }
  return false;
}

Seg to_seg(const ShmSeg& g) {
  Seg s;
  s.rows = static_cast<int>(g.rows);
  s.dim = static_cast<int>(g.dim);
  s.start = g.start;
  s.now0 = g.now0;
  s.t0 = g.t0;
  s.t1 = g.t1;
  std::memcpy(s.knots, g.knots, sizeof(s.knots));
  std::memcpy(s.ctrl, g.ctrl, sizeof(s.ctrl));
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const char* shm_path = "/dev/shm/bsp_ctl";
  const char* port = nullptr;
  std::string calib;
  double hz = 50.0;
  int read_every = 1;
  bool rt = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--shm") shm_path = next();
    else if (a == "--port") port = next();
    else if (a == "--calib") calib = next();
    else if (a == "--hz") hz = atof(next());
    else if (a == "--read-every") read_every = std::max(1, atoi(next()));
    else if (a == "--rt") rt = true;
  }
  if (rt) {
    sched_param sp{};
    sp.sched_priority = 50;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
      perror("SCHED_FIFO (needs CAP_SYS_NICE); normal priority");
    }
  }
  Shm* shm = map_shm(shm_path);
  Bus bus;
  std::vector<Motor> motors;
  if (port) {
    if (calib.empty()) {
      fprintf(stderr, "--port needs --calib\n");
      return 2;
    }
    motors = read_calib(calib);
    if (!bus.open_port(port)) {
      perror(port);
      return 2;
    }
  }
  const int dim = kDim;
  ShmIn in;
  Seg cur, old;
  double last[kMaxDim];
  bool have_last = false;
  int64_t seg_id = -1, old_id = -1;
  const int64_t period_ns = static_cast<int64_t>(1e9 / hz);
  timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);
  double prev_tick = now_s();
  int64_t tick = 0;
  shm->out.pid = getpid();
  shm->out.alive = 1;

  while (true) {
    const double t_tick = now_s();
    if (!read_input(shm, &in)) in.mode = kHold;
    if (in.mode == kQuit) break;

    double bus_ms = 0, read_ms = 0;
    if (port && tick % read_every == 0) {
      double q[kMaxDim];
      const double t = now_s();
      const bool ok = bus.read_present(motors, q);
      read_ms = (now_s() - t) * 1e3;
      if (ok) {
        shm->out.seq += 1;
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(shm->out.state, q, sizeof(double) * dim);
        shm->out.state_time = now_s();
        std::atomic_thread_fence(std::memory_order_release);
        shm->out.seq += 1;
      } else {
        shm->out.read_errors += 1;
      }
    }

    double target[kMaxDim];
    bool have_target = false;
    if (in.mode == kRun && in.seg.seg_id >= 0) {
      if (in.seg.seg_id != seg_id) {
        seg_id = in.seg.seg_id;
        cur = to_seg(in.seg);
      }
      if (in.fade_on && in.old.seg_id != old_id) {
        old_id = in.old.seg_id;
        old = to_seg(in.old);
      }
      const double now = now_s();
      const double tau = (now - in.t_start) * in.fps * in.speed;
      const double s = std::clamp(cur.start + tau - cur.now0, cur.t0,
                                  cur.t1 - 1e-6);
      spline_eval(cur, static_cast<int>(in.degree), s, target);
      if (in.fade_on && now < in.fade_t1) {
        double o[kMaxDim];
        const double os = std::clamp(old.start + tau - old.now0, old.t0,
                                     old.t1 - 1e-6);
        spline_eval(old, static_cast<int>(in.degree), os, o);
        double a = (now - in.fade_t0) / (in.fade_t1 - in.fade_t0);
        a = a * a * (3 - 2 * a);
        for (int d = 0; d < dim; ++d) {
          target[d] = (1 - a) * o[d] + a * target[d];
        }
      }
      have_target = true;
    } else if (in.mode == kDirect) {
      std::memcpy(target, in.direct, sizeof(double) * dim);
      have_target = true;
    } else {
      have_last = false;  // hold: the next run starts from its curve
    }

    if (have_target) {
      if (have_last) {
        for (int d = 0; d < dim; ++d) {
          target[d] = last[d] + std::clamp(target[d] - last[d],
                                           -in.max_step, in.max_step);
        }
      }
      bool finite = true;
      for (int d = 0; d < dim; ++d) finite &= std::isfinite(target[d]);
      if (finite) {
        std::memcpy(last, target, sizeof(double) * dim);
        have_last = true;
        const double t = now_s();
        if (port) bus.write_goal(motors, target);
        bus_ms = (now_s() - t) * 1e3;
        shm->out.seq += 1;
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(shm->out.last_cmd, target, sizeof(double) * dim);
        shm->out.cmd_time = now_s();
        shm->out.cmd_seg = static_cast<double>(seg_id);
        std::atomic_thread_fence(std::memory_order_release);
        shm->out.seq += 1;
      }
    }

    // tick log ring: tick start, period, bus write ms, bus read ms, cmd
    const int64_t li = tick % kLogN;
    for (int d = 0; d < kMaxDim; ++d) {
      shm->log[li][4 + d] = (have_target && d < dim) ? last[d] : NAN;
    }
    shm->log[li][0] = t_tick;
    shm->log[li][1] = (t_tick - prev_tick) * 1e3;
    shm->log[li][2] = bus_ms;
    shm->log[li][3] = read_ms;
    prev_tick = t_tick;
    ++tick;
    shm->out.n_ticks = static_cast<double>(tick);

    next.tv_nsec += period_ns;
    while (next.tv_nsec >= 1000000000L) {
      next.tv_nsec -= 1000000000L;
      next.tv_sec += 1;
    }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
  }
  shm->out.alive = 0;
  bus.close_port();
  return 0;
}
