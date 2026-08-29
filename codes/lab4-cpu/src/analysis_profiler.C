#include "analysis_profiler.h"

#ifdef AMSS_ENABLE_ANALYSIS_PROFILE

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

struct ProfileState {
  bool active;
  int step;
  int rank;
  double physical_time;
  double radius;
  unsigned long sequence;
  char phase[96];
  std::string run_id;
  FILE *output;

  ProfileState()
      : active(false), step(-1), rank(-1), physical_time(0.0), radius(-1.0),
        sequence(0), output(NULL) {
    phase[0] = '\0';
  }
};

ProfileState state;

std::string env_or_default(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  return value && value[0] ? value : fallback;
}

void ensure_output() {
  if (state.output)
    return;

  PMPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
  std::ostringstream default_run;
  default_run << "pid" << static_cast<long>(getpid());
  state.run_id = env_or_default("AMSS_ANALYSIS_PROFILE_RUN_ID", default_run.str());
  const std::string output_dir = env_or_default("AMSS_ANALYSIS_PROFILE_DIR", ".");

  std::ostringstream filename;
  filename << output_dir << "/analysis_profile_run-" << state.run_id
           << "_rank-" << std::setw(5) << std::setfill('0') << state.rank << ".tsv";
  state.output = std::fopen(filename.str().c_str(), "w");
  if (!state.output) {
    std::fprintf(stderr, "analysis profiler: cannot open %s\n", filename.str().c_str());
    PMPI_Abort(MPI_COMM_WORLD, 97);
  }
  std::fprintf(state.output,
               "run_id\tstep\ttime\trank\tphase\tradius\tevent\tcollective\tsequence\tcount\tbytes\tcomm_size\tduration_s\n");
  std::fflush(state.output);
}

void write_event(const char *phase, double radius, const char *event,
                 const char *collective, unsigned long sequence,
                 long long count, long long bytes, int comm_size,
                 double duration_s) {
  ensure_output();
  std::fprintf(state.output,
               "%s\t%d\t%.17g\t%d\t%s\t%.17g\t%s\t%s\t%lu\t%lld\t%lld\t%d\t%.17g\n",
               state.run_id.c_str(), state.step, state.physical_time, state.rank,
               phase ? phase : "", radius, event ? event : "",
               collective ? collective : "", sequence, count, bytes,
               comm_size, duration_s);
}

} // namespace

double analysis_profile_now() { return PMPI_Wtime(); }

void analysis_profile_begin(int step, double physical_time) {
  ensure_output();
  state.step = step;
  state.physical_time = physical_time;
  state.radius = -1.0;
  state.sequence = 0;
  std::strncpy(state.phase, "AnalysisStuff", sizeof(state.phase) - 1);
  state.phase[sizeof(state.phase) - 1] = '\0';
  state.active = true;
}

void analysis_profile_end(double start_time) {
  const double duration = PMPI_Wtime() - start_time;
  write_event("AnalysisStuff", -1.0, "wall", "", 0, 0, 0, 0, duration);
  std::fflush(state.output);
  state.active = false;
}

void analysis_profile_set_phase(const char *phase, double radius) {
  if (!state.active)
    return;
  std::strncpy(state.phase, phase ? phase : "", sizeof(state.phase) - 1);
  state.phase[sizeof(state.phase) - 1] = '\0';
  state.radius = radius;
}

void analysis_profile_record(const char *phase, double radius,
                             const char *event, double duration_s) {
  if (!state.active)
    return;
  write_event(phase, radius, event, "", 0, 0, 0, 0, duration_s);
}

extern "C" int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (!state.active)
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

  int type_size = 0;
  int comm_size = 0;
  PMPI_Type_size(datatype, &type_size);
  PMPI_Comm_size(comm, &comm_size);
  const double start = PMPI_Wtime();
  const int result = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  const double duration = PMPI_Wtime() - start;
  write_event(state.phase, state.radius, "call", "MPI_Allreduce",
              ++state.sequence, count,
              static_cast<long long>(count) * type_size, comm_size, duration);
  return result;
}

#endif
