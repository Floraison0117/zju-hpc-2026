#ifndef TRSM_SME_UPDATE_H
#define TRSM_SME_UPDATE_H

#include <stddef.h>

/*
 * Update one complete row-major tile in place:
 *
 *     C[16,32] -= A_pack[K,16] * B_pack[K,32]
 *
 * A_pack and B_pack are K-major.  The assembly implementation is intended
 * for 64-byte aligned packed buffers and accepts an arbitrary byte stride for
 * C.  It does not modify A_pack or B_pack.
 */
void sme_update_16x32(const double *a_pack, const double *b_pack,
                      double *c, int k, int ldc_bytes);

/* Experimental variant: initialize ZA from C, then apply FMOPS in place. */
void sme_update_16x32_za_init(const double *a_pack, const double *b_pack,
                              double *c, int k, int ldc_bytes);

/* Experimental direct ZA memory form: load C with ZA load instructions,
 * apply FMOPS, and store ZA directly back to C. */
void sme_update_16x32_za_ldst(const double *a_pack, const double *b_pack,
                              double *c, int k, int ldc_bytes);

/*
 * Persistent variant.  The same packed A tile is applied to nt adjacent
 * packed B tiles and adjacent C tiles while SME remains enabled.  b_pack is
 * laid out as nt consecutive [K,32] tiles.  C tiles are separated by 256
 * bytes and rows within a tile use ldc_bytes.
 */
void sme_update_16x32_batch(const double *a_pack, const double *b_pack,
                            double *c, int k, int ldc_bytes, int nt);

/*
 * Persistent row-oriented variant.  A_pack contains mt consecutive
 * [K,16] tiles, b_pack contains one [K,32] tile, and C tiles begin every
 * 16 rows.  SME is entered once for the complete row-tile sequence.
 */
void sme_update_16x32_rows_batch(const double *a_pack, const double *b_pack,
                                 double *c, int k, int ldc_bytes, int mt);

/* Persistent two-dimensional variant for a complete full-tile row/column
 * grid.  A_pack has mt consecutive tiles and b_pack has nt consecutive
 * tiles; C is row-major with ldc_bytes between rows. */
void sme_update_16x32_grid_batch(const double *a_pack, const double *b_pack,
                                 double *c, int k, int ldc_bytes,
                                 int nt, int mt);

/* Experimental grid variant that initializes each ZA accumulator from C. */
void sme_update_16x32_za_init_grid_batch(const double *a_pack,
                                         const double *b_pack, double *c,
                                         int k, int ldc_bytes,
                                         int nt, int mt);

/* Apply a packed A tile to a raw row-major B panel with a byte row stride. */
void sme_update_16x32_strided_b(const double *a_pack, const double *b,
                                int k, int b_ld_bytes, double *c,
                                int ldc_bytes);

/* Persistent raw-B form.  The row-tile grid is processed by one SME
 * region; a_pack has mt consecutive [K,16] tiles and b is reused for every
 * row tile. */
void sme_update_16x32_strided_b_grid(const double *a_pack, const double *b,
                                     int k, int b_ld_bytes, double *c,
                                     int ldc_bytes, int nt, int mt);

/* Persistent task-list form.  Each list entry names one complete 16x32
 * tile; the worker keeps SME enabled while processing count entries. */
void sme_update_16x32_ptr_batch(const double *const *a_tiles,
                                const double *const *b_tiles,
                                double *const *c_tiles, int k,
                                int ldc_bytes, int count);

/*
 * Correctness-preserving dispatcher for a packed tile.  Full 16x32 tiles use
 * the SME kernel; all tails, zero-sized inputs, and unsupported shapes use a
 * scalar fallback.  a_stride and b_stride are measured in doubles, while
 * ldc_bytes is measured in bytes.
 */
void sme_update_tile(const double *a_pack, int a_stride,
                     const double *b_pack, int b_stride,
                     double *c, int ldc_bytes,
                     int m, int n, int k);

/* Scalar reference used by tests and by callers that intentionally disable
 * the assembly path. */
void sme_update_tile_fallback(const double *a_pack, int a_stride,
                              const double *b_pack, int b_stride,
                              double *c, int ldc_bytes,
                              int m, int n, int k);

#endif
