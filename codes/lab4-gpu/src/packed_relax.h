#ifndef AMSS_PACKED_RELAX_H
#define AMSS_PACKED_RELAX_H

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace amss_packed_relax
{
struct Direction
{
  std::vector<double> diagonal;
  std::vector<double> lower;
  std::vector<double> upper;
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<double> values;
};

inline int index(int ivar, int i, int j, int k,
                 int nvar, int n1, int n2, int n3)
{
  if (i < 0)
    i = -(i + 1);
  if (i >= n1)
    i = 2 * n1 - (i + 1);
  if (j < 0)
    j = -(j + 1);
  if (j >= n2)
    j = 2 * n2 - (j + 1);
  if (k < 0)
    k += n3;
  if (k >= n3)
    k -= n3;
  return ivar + nvar * (i + n1 * (j + n2 * k));
}

inline void fail_row(const char *direction, int row, const char *reason)
{
  std::fprintf(stderr, "packed relax invariant failed: direction=%s row=%d %s\n",
               direction, row, reason);
  std::abort();
}

inline void build_direction(Direction &out, bool along_be,
                            int nvar, int n1, int n2, int n3,
                            const int *ncols, int **cols, double **matrix)
{
  const int ntotal = nvar * n1 * n2 * n3;
  out.diagonal.assign(ntotal, 0.0);
  out.lower.assign(ntotal, 0.0);
  out.upper.assign(ntotal, 0.0);
  out.row_offsets.resize(ntotal + 1);
  out.columns.clear();
  out.values.clear();
  out.columns.reserve(static_cast<std::size_t>(ntotal) * 16);
  out.values.reserve(static_cast<std::size_t>(ntotal) * 16);

  for (int row = 0; row < ntotal; ++row)
  {
    int cell = row / nvar;
    const int ivar = row - cell * nvar;
    const int i = cell % n1;
    cell /= n1;
    const int j = cell % n2;
    const int k = cell / n2;
    const int position = along_be ? j : i;
    const int length = along_be ? n2 : n1;
    const int center = row;
    const int minus = along_be
                          ? index(ivar, i, j - 1, k, nvar, n1, n2, n3)
                          : index(ivar, i - 1, j, k, nvar, n1, n2, n3);
    const int plus = along_be
                         ? index(ivar, i, j + 1, k, nvar, n1, n2, n3)
                         : index(ivar, i + 1, j, k, nvar, n1, n2, n3);
    bool have_diagonal = false;
    bool have_lower = (position == 0);
    bool have_upper = (position == length - 1);
    out.row_offsets[row] = static_cast<int>(out.columns.size());

    for (int entry = 0; entry < ncols[row]; ++entry)
    {
      const int column = cols[row][entry];
      const double value = matrix[row][entry];
      if (column != minus && column != center && column != plus)
      {
        out.columns.push_back(column);
        out.values.push_back(value);
      }
      else
      {
        if (column == minus && position > 0)
        {
          out.lower[row] = value;
          have_lower = true;
        }
        if (column == center)
        {
          out.diagonal[row] = value;
          have_diagonal = true;
        }
        if (column == plus && position < length - 1)
        {
          out.upper[row] = value;
          have_upper = true;
        }
      }
    }

    if (!have_diagonal)
      fail_row(along_be ? "be" : "al", row, "missing diagonal");
    if (!have_lower)
      fail_row(along_be ? "be" : "al", row, "missing lower diagonal");
    if (!have_upper)
      fail_row(along_be ? "be" : "al", row, "missing upper diagonal");
    out.row_offsets[row + 1] = static_cast<int>(out.columns.size());
  }
}

struct Storage
{
  Direction be;
  Direction al;
  double build_seconds;
  std::size_t bytes;

  Storage(int nvar, int n1, int n2, int n3,
          const int *ncols, int **cols, double **matrix)
      : build_seconds(0.0), bytes(0)
  {
    const auto begin = std::chrono::steady_clock::now();
    build_direction(be, true, nvar, n1, n2, n3, ncols, cols, matrix);
    build_direction(al, false, nvar, n1, n2, n3, ncols, cols, matrix);
    build_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - begin)
                        .count();
    bytes = (be.diagonal.size() + be.lower.size() + be.upper.size() +
             be.values.size() + al.diagonal.size() + al.lower.size() +
             al.upper.size() + al.values.size()) * sizeof(double) +
            (be.row_offsets.size() + be.columns.size() +
             al.row_offsets.size() + al.columns.size()) * sizeof(int);
    if (std::getenv("AMSS_PACKED_RELAX_DIAGNOSTICS") != nullptr)
      std::fprintf(stderr,
                   "PACKED_RELAX build_s=%.9f bytes=%zu be_offdiag=%zu al_offdiag=%zu\n",
                   build_seconds, bytes, be.values.size(), al.values.size());
  }
};

inline const Storage *&active_slot()
{
  static const Storage *active = nullptr;
  return active;
}

inline const Storage *active()
{
  return active_slot();
}

struct Scope
{
  explicit Scope(const Storage *storage) { active_slot() = storage; }
  ~Scope() { active_slot() = nullptr; }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
};
}

#endif

