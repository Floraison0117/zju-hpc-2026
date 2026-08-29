#ifndef AMSS_ANALYSIS_PROFILER_H
#define AMSS_ANALYSIS_PROFILER_H

#ifdef AMSS_ENABLE_ANALYSIS_PROFILE

double analysis_profile_now();
void analysis_profile_begin(int step, double physical_time);
void analysis_profile_end(double start_time);
void analysis_profile_set_phase(const char *phase, double radius);
void analysis_profile_record(const char *phase, double radius,
                             const char *event, double duration_s);

#else

inline double analysis_profile_now() { return 0.0; }
inline void analysis_profile_begin(int, double) {}
inline void analysis_profile_end(double) {}
inline void analysis_profile_set_phase(const char *, double) {}
inline void analysis_profile_record(const char *, double, const char *, double) {}

#endif

#endif
