// Shared memory layout of bsp_ctl. All fields are float64, so Python
// (bsp/cpp_control.py) maps the same file as one numpy array. Keep the
// field order in sync with LAYOUT in cpp_control.py.
#pragma once

constexpr int kMaxRows = 32;
constexpr int kMaxDim = 8;
constexpr int kMaxDeg = 5;
constexpr int kDim = 6;
constexpr int kLogN = 8192;
constexpr int kLogCols = 4 + kMaxDim;  // t, period, write, read, cmd

enum Mode { kHold = 0, kRun = 1, kDirect = 2, kQuit = 3 };

struct ShmSeg {
  double seg_id, rows, dim, start, now0, t0, t1;
  double knots[kMaxRows];
  double ctrl[kMaxRows * kMaxDim];
};

// written by Python under seq (odd while writing)
struct ShmIn {
  double seq, mode, t_start, fps, speed, max_step, degree;
  double fade_on, fade_t0, fade_t1;
  double direct[kMaxDim];
  ShmSeg seg;
  ShmSeg old;
};

// written by bsp_ctl under seq
struct ShmOut {
  double seq, alive, pid, n_ticks, read_errors;
  double state_time, cmd_time, cmd_seg;
  double state[kMaxDim];
  double last_cmd[kMaxDim];
};

struct Shm {
  ShmIn in;
  ShmOut out;
  double log[kLogN][kLogCols];
};

static_assert(sizeof(ShmSeg) == 8 * (7 + kMaxRows + kMaxRows * kMaxDim));
static_assert(sizeof(ShmIn) ==
              8 * (10 + kMaxDim) + 2 * sizeof(ShmSeg));
static_assert(sizeof(ShmOut) == 8 * (8 + 2 * kMaxDim));
