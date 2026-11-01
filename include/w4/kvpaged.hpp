#pragma once

// Paged W4 KV cache (R7), host side: the R5 slab format with one level of
// indirection. The physical pool is, per head, npages pages of `page`
// positions each; a sequence's block table maps logical pages to physical
// ones. Paging changes where a position lives, never what it contains —
// the dequant-at-load craft and every gate carry over unchanged.
//
// The point of paging in production is prefix sharing: two sequences whose
// block tables reference the same physical pages share that prefix's
// memory exactly once. The builder here takes a contiguous cache plus a
// physical permutation (shuffled placement is the worst case, so that is
// what the gates use), and the two-sequence driver demonstrates the share.

#include <cuda_fp16.h>

#include <numeric>
#include <stdexcept>
#include <vector>

#include "kvcache.hpp"

namespace w4::kv {

struct KVCacheW4Paged {
  int H = 0, S = 0, D = 0, gs = 0, G = 0;
  int page = 0;    // positions per page
  int npages = 0;  // physical pages in the pool (per head)
  bool symmetric = true;
  std::vector<uint32_t> q_k, q_v;  // H slabs of npages*page*words(D)
  std::vector<__half> s_k, s_v;    // H slabs of npages*page*G
  std::vector<uint32_t> z_k, z_v;  // asymmetric only
  std::vector<int> block_table;    // ceil(S/page) entries, values < npages
};

inline void validate_paged(const KVCacheW4Paged& c) {
  if (c.H < 1 || c.S < 1 || c.D < 1)
    throw std::invalid_argument("paged: bad dims");
  if (c.gs < 8 || c.gs % 8 != 0)
    throw std::invalid_argument("paged: gs must be a multiple of 8");
  if (c.G != (c.D + c.gs - 1) / c.gs)
    throw std::invalid_argument("paged: G inconsistent with D, gs");
  if (c.page < 1 || c.npages < 1)
    throw std::invalid_argument("paged: bad page geometry");
  const size_t lpages = (size_t)(c.S + c.page - 1) / c.page;
  if (c.block_table.size() != lpages)
    throw std::invalid_argument("paged: block table size");
  if ((size_t)c.npages * c.page < lpages * c.page)
    throw std::invalid_argument("paged: pool smaller than table");
  const size_t qw = (size_t)c.H * c.npages * c.page * words(c.D);
  const size_t sw = (size_t)c.H * c.npages * c.page * c.G;
  if (c.q_k.size() != qw || c.q_v.size() != qw)
    throw std::invalid_argument("paged: q size");
  if (c.s_k.size() != sw || c.s_v.size() != sw)
    throw std::invalid_argument("paged: scales size");
  if (!c.symmetric &&
      (c.z_k.size() != (size_t)c.H * c.npages * c.page * words(c.G) ||
       c.z_v.size() != (size_t)c.H * c.npages * c.page * words(c.G)))
    throw std::invalid_argument("paged: zeros size");
  std::vector<char> seen((size_t)c.npages, 0);
  for (const int b : c.block_table) {
    if (b < 0 || b >= c.npages)
      throw std::invalid_argument("paged: block table entry out of pool");
    if (seen[(size_t)b]++)
      throw std::invalid_argument("paged: block table repeats a page");
  }
}

// Scatter a contiguous cache into the pool by the block table: logical
// page i lands at physical page block_table[i]. Positions past S in the
// last logical page are zero-filled.
inline KVCacheW4Paged paged_from_contiguous(const KVCacheW4& c, int page,
                                            const std::vector<int>& bt) {
  validate_cache(c);
  const size_t lpages = (size_t)(c.S + page - 1) / page;
  if (page < 1) throw std::invalid_argument("paged_from: bad page size");
  if (bt.size() != lpages)
    throw std::invalid_argument("paged_from: block table size");
  std::vector<char> seen(bt.size(), 0);
  for (const int b : bt) {
    if (b < 0 || b >= (int)bt.size() || seen[(size_t)b]++)
      throw std::invalid_argument("paged_from: bad block table");
  }
  KVCacheW4Paged p;
  p.H = c.H;
  p.S = c.S;
  p.D = c.D;
  p.gs = c.gs;
  p.G = c.G;
  p.page = page;
  p.npages = (int)bt.size();
  p.symmetric = c.symmetric;
  p.block_table = bt;
  const size_t wq = (size_t)p.H * p.npages * page * words(c.D);
  const size_t ws = (size_t)p.H * p.npages * page * p.G;
  const size_t wz = (size_t)p.H * p.npages * page * words(p.G);
  p.q_k.assign(wq, 0);
  p.q_v.assign(wq, 0);
  p.s_k.assign(ws, __float2half_rn(0.0f));
  p.s_v.assign(ws, __float2half_rn(0.0f));
  if (!c.symmetric) {
    p.z_k.assign(wz, 0);
    p.z_v.assign(wz, 0);
  }
  const int wd = words(c.D);
  for (int h = 0; h < c.H; ++h) {
    const size_t src_q = (size_t)h * c.S * wd;
    const size_t src_s = (size_t)h * c.S * c.G;
    const size_t src_z = (size_t)h * c.S * words(c.G);
    for (size_t lp = 0; lp < lpages; ++lp) {
      const int phys = bt[lp];
      for (int j = 0; j < page; ++j) {
        const int t = (int)lp * page + j;
        if (t >= c.S) break;
        const size_t dst_row = (size_t)phys * page + j;
        const size_t dq = ((size_t)h * p.npages * page + dst_row) * wd;
        const size_t ds = ((size_t)h * p.npages * page + dst_row) * p.G;
        const size_t dz = ((size_t)h * p.npages * page + dst_row) * words(p.G);
        const size_t sq = src_q + (size_t)t * wd;
        const size_t ss = src_s + (size_t)t * c.G;
        std::copy_n(c.q_k.begin() + sq, wd, p.q_k.begin() + dq);
        std::copy_n(c.q_v.begin() + sq, wd, p.q_v.begin() + dq);
        std::copy_n(c.s_k.begin() + ss, c.G, p.s_k.begin() + ds);
        std::copy_n(c.s_v.begin() + ss, c.G, p.s_v.begin() + ds);
        if (!c.symmetric) {
          const size_t sz = src_z + (size_t)t * words(c.G);
          std::copy_n(c.z_k.begin() + sz, words(c.G), p.z_k.begin() + dz);
          std::copy_n(c.z_v.begin() + sz, words(c.G), p.z_v.begin() + dz);
        }
      }
    }
  }
  return p;
}

}  // namespace w4::kv
