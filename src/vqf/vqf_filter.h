/*
 * ============================================================================
 *
 *       Filename:  vqf_filter.h
 *
 *         Author:  Prashant Pandey (), ppandey@berkeley.edu
 *   Organization: 	LBNL/UCB
 *
 * ============================================================================
 */

#pragma once
#ifndef _VQF_FILTER_H_
#define _VQF_FILTER_H_

#include <inttypes.h>
#include <stdbool.h>

#ifdef __cplusplus
#define restrict __restrict__
#else
#error This is C++ only!
#endif

// metadata: 1 --> end of the run
// Each 1 is preceded by k 0s, where k is the number of remainders in that
// run.

template <int TAG_BITS>
struct vqf_block;

//=#=#==#==#===============+=+=+=+=++=++++++++++++++-++-+--+-+----+---------------
// We are using 8-bit tags.
// One block consists of 48 8-bit slots covering 80 buckets, and 80+48 = 128
// bits of metadata.
//
template <>
struct __attribute__((__packed__)) vqf_block<8> {
   uint64_t md[2];
   uint8_t tags[48];
};

//=#=#==#==#===============+=+=+=+=++=++++++++++++++-++-+--+-+----+---------------
// We are using 12-bit tags.
// One block consists of 32 12-bit slots covering 96 buckets, and 96+32 = 128
// bits of metadata.
// NOTE: not supported yet.
//
template <>
struct __attribute__((__packed__)) vqf_block<12> {
   uint64_t md[2];
   uint8_t tags[32];  // 32 12-bit tags
};

//=#=#==#==#===============+=+=+=+=++=++++++++++++++-++-+--+-+----+---------------
// We are using 16-bit tags.
// One block consists of 28 16-bit slots covering 36 buckets, and 36+28 = 64
// bits of metadata.
//
template <>
struct __attribute__((__packed__)) vqf_block<16> {
   uint64_t md;
   uint16_t tags[28];
};

//=#=#==#==#===============+=+=+=+=++=++++++++++++++-++-+--+-+----+---------------
//
struct vqf_metadata {
   uint64_t total_size_in_bytes;
   uint64_t key_remainder_bits;
   uint64_t range;
   uint64_t nblocks;
   uint64_t nelts;
   uint64_t nslots;
};

template <int TAG_BITS>
struct vqf_filter {
   vqf_metadata metadata;
   vqf_block<TAG_BITS> blocks[];
};

/** \brief Returns the number of bytes required to initialize a filter with the specified number of slots.
 */
template <int TAG_BITS>
uint64_t vqf_required_size(uint64_t nslots);

/** \brief Allocates, initializes, and returns a filter with the specified number of slots.
 */
template <int TAG_BITS>
[[nodiscard]] vqf_filter<TAG_BITS>* vqf_init(uint64_t nslots);

/** \brief Initializes a pre-allocated filter.
 *
 * The memory pointed to by `filter` MUST be at least `vqf_required_size(nslots)` bytes large.
 *
 * \return `filter`
 */
template <int TAG_BITS>
vqf_filter<TAG_BITS>* vqf_init_in_place(vqf_filter<TAG_BITS>* restrict filter, uint64_t nslots);

/** \brief Adds the given hash value to the filter.
 */
template <int TAG_BITS>
bool vqf_insert(vqf_filter<TAG_BITS>* restrict filter, uint64_t hash);

/** \brief Removes the given hash value from the filter.
 */
template <int TAG_BITS>
bool vqf_remove(vqf_filter<TAG_BITS>* restrict filter, uint64_t hash);

/** \brief Tests the filter to see whether it _might_ contain the given hash.
 */
template <int TAG_BITS>
bool vqf_is_present(vqf_filter<TAG_BITS>* restrict filter, uint64_t hash);

/** \brief Returns the in-memory size of the filter, in bytes.
 */
template <int TAG_BITS>
uint64_t vqf_filter_size(vqf_filter<TAG_BITS>* restrict filter);

/** \brief Frees `*filter` (which MUST have been allocated via vqf_init) and sets the pointer to null.
 */
template <int TAG_BITS>
void vqf_free(vqf_filter<TAG_BITS>** restrict filter);

#endif  // _VQF_FILTER_H_
