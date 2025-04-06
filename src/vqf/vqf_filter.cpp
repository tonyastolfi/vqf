/*
 * ============================================================================
 *
 *       Filename:  vqf_filter.c
 *
 *         Author:  Prashant Pandey (), ppandey@berkeley.edu
 *   Organization:  LBNL/UCB
 *
 * ============================================================================
 */

#include <assert.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <algorithm>
#include <iostream>

#include "vqf_filter.h"
#include "vqf_precompute.h"

// ALT block check is set of 75% of the number of slots
//
template <int TAG_BITS>
struct vqf_constants;

template <>
struct vqf_constants<8> {
   static constexpr uint64_t TAG_MASK = 0xff;
   static constexpr uint64_t QUQU_SLOTS_PER_BLOCK = 48;
   static constexpr uint64_t QUQU_BUCKETS_PER_BLOCK = 80;
   static constexpr uint64_t QUQU_CHECK_ALT = 92;
};

template <>
struct vqf_constants<12> {
   static constexpr uint64_t TAG_MASK = 0xfff;
   static constexpr uint64_t QUQU_SLOTS_PER_BLOCK = 32;
   static constexpr uint64_t QUQU_BUCKETS_PER_BLOCK = 96;
   static constexpr uint64_t QUQU_CHECK_ALT = 104;
};

template <>
struct vqf_constants<16> {
   static constexpr uint64_t TAG_MASK = 0xffff;
   static constexpr uint64_t QUQU_SLOTS_PER_BLOCK = 28;
   static constexpr uint64_t QUQU_BUCKETS_PER_BLOCK = 36;
   static constexpr uint64_t QUQU_CHECK_ALT = 43;
};

#ifdef __AVX512BW__
extern "C" {
extern __m512i SHUFFLE[];
extern __m512i SHUFFLE_REMOVE[];
extern __m512i SHUFFLE16[];
extern __m512i SHUFFLE_REMOVE16[];
}
#endif

#define LOCK_MASK (1ULL << 63)
#define UNLOCK_MASK ~(1ULL << 63)

static inline void lock(vqf_block<8>& block)
{
#ifdef ENABLE_THREADS
   uint64_t* data = block.md + 1;

   while ((__sync_fetch_and_or(data, LOCK_MASK) & (1ULL << 63)) != 0) {
      continue;
   }
#endif
}

static inline void lock(vqf_block<16>& block)
{
#ifdef ENABLE_THREADS
   uint64_t* data = &block.md;

   while ((__sync_fetch_and_or(data, LOCK_MASK) & (1ULL << 63)) != 0) {
      continue;
   }
#endif
}

static inline void unlock(vqf_block<8>& block)
{
#ifdef ENABLE_THREADS
   uint64_t* data = block.md + 1;

   __sync_fetch_and_and(data, UNLOCK_MASK);
#endif
}

static inline void unlock(vqf_block<16>& block)
{
#ifdef ENABLE_THREADS
   uint64_t* data = &block.md;

   __sync_fetch_and_and(data, UNLOCK_MASK);
#endif
}

template <int TAG_BITS>
static inline void lock_blocks(vqf_filter<TAG_BITS>* restrict filter, uint64_t index1, uint64_t index2)
{
#ifdef ENABLE_THREADS
   if (index1 < index2) {
      lock(filter->blocks[index1 / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
      lock(filter->blocks[index2 / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
   } else {
      lock(filter->blocks[index2 / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
      lock(filter->blocks[index1 / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
   }
#endif
}

template <int TAG_BITS>
static inline void unlock_blocks(vqf_filter<TAG_BITS>* restrict filter, uint64_t index1, uint64_t index2)
{
#ifdef ENABLE_THREADS
   if (index1 < index2) {
      unlock(filter->blocks[index1 / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
      unlock(filter->blocks[index2 / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
   } else {
      unlock(filter->blocks[index2 / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
      unlock(filter->blocks[index1 / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
   }
#endif
}

static inline int word_rank(uint64_t val)
{
   return __builtin_popcountll(val);
}

// Returns the position of the rank'th 1.  (rank = 0 returns the 1st 1)
// Returns 64 if there are fewer than rank+1 1s.
static inline uint64_t word_select(uint64_t val, int rank)
{
   val = _pdep_u64(one[rank], val);
   return _tzcnt_u64(val);
}

// select(vec, 0) -> -1
// select(vec, i) -> 128, if i > popcnt(vec)
static inline int64_t select_128_old(__uint128_t vector, uint64_t rank)
{
   uint64_t lower_word = vector & 0xffffffffffffffff;
   uint64_t lower_pdep = _pdep_u64(one[rank], lower_word);
   //uint64_t lower_select = word_select(lower_word, rank);
   if (lower_pdep != 0) {
      //assert(rank < word_rank(lower_word));
      return _tzcnt_u64(lower_pdep);
   }
   rank = rank - word_rank(lower_word);
   uint64_t higher_word = vector >> 64;
   return word_select(higher_word, rank) + 64;
}

static inline uint64_t lookup_64(uint64_t vector, uint64_t rank)
{
   uint64_t lower_return = _pdep_u64(one[rank], vector) >> rank << (sizeof(uint64_t) / 2);
   return lower_return;
}

static inline uint64_t lookup_128(const uint64_t* vector, uint64_t rank)
{
   uint64_t lower_word = vector[0];
   uint64_t lower_rank = word_rank(lower_word);
   uint64_t lower_return = _pdep_u64(one[rank], lower_word) >> rank << sizeof(__uint128_t);
   int64_t higher_rank = (int64_t)rank - lower_rank;
   uint64_t higher_word = vector[1];
   uint64_t higher_return = _pdep_u64(one[higher_rank], higher_word);
   higher_return <<= (64 + sizeof(__uint128_t) - rank);
   return lower_return + higher_return;
}

static inline int64_t select_64(uint64_t vector, uint64_t rank)
{
   return _tzcnt_u64(lookup_64(vector, rank));
}

static inline int64_t select_128(const uint64_t* vector, uint64_t rank)
{
   return _tzcnt_u64(lookup_128(vector, rank));
}

template <int TAG_BITS>
struct vqf_print;

// assumes little endian

template <>
struct vqf_print<8> {
   static constexpr int TAG_BITS = 8;

   static void print_bits(__uint128_t num, int numbits)
   {
      int i;
      for (i = 0; i < numbits; i++) {
         if (i != 0 && i % 8 == 0) {
            printf(":");
         }
         printf("%d", ((num >> i) & 1) == 1);
      }
      puts("");
   }

   static void print_tags(const uint8_t* tags, uint32_t size)
   {
      for (uint8_t i = 0; i < size; i++)
         printf("%d ", (uint32_t)tags[i]);
      printf("\n");
   }

   static void print_block(const vqf_filter<TAG_BITS>* filter, uint64_t block_index)
   {
      printf("block index: %ld\n", block_index);
      printf("metadata: ");
      const uint64_t* md = filter->blocks[block_index].md;
      print_bits(*(__uint128_t*)md, vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK +
                                        vqf_constants<TAG_BITS>::QUQU_SLOTS_PER_BLOCK);
      printf("tags: ");
      print_tags(filter->blocks[block_index].tags, vqf_constants<TAG_BITS>::QUQU_SLOTS_PER_BLOCK);
   }
};

template <>
struct vqf_print<16> {
   static constexpr int TAG_BITS = 16;

   static void print_bits(uint64_t num, int numbits)
   {
      int i;
      for (i = 0; i < numbits; i++) {
         if (i != 0 && i % 8 == 0) {
            printf(":");
         }
         printf("%d", ((num >> i) & 1) == 1);
      }
      puts("");
   }

   static void print_tags(const uint16_t* tags, uint32_t size)
   {
      for (uint8_t i = 0; i < size; i++)
         printf("%d ", (uint32_t)tags[i]);
      printf("\n");
   }

   static void print_block(const vqf_filter<TAG_BITS>* filter, uint64_t block_index)
   {
      printf("block index: %ld\n", block_index);
      printf("metadata: ");
      uint64_t md = filter->blocks[block_index].md;
      print_bits(md, vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK +
                         vqf_constants<TAG_BITS>::QUQU_SLOTS_PER_BLOCK);
      printf("tags: ");
      print_tags(filter->blocks[block_index].tags, vqf_constants<TAG_BITS>::QUQU_SLOTS_PER_BLOCK);
   }
};

#ifdef __AVX512BW__

static inline void update_tags_512(vqf_block<8>* restrict block, uint8_t index, uint8_t tag)
{
   block->tags[47] = tag;  // add tag at the end

   __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
   vector = _mm512_permutexvar_epi8(SHUFFLE[index], vector);
   _mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}

static inline void remove_tags_512(vqf_block<8>* restrict block, uint8_t index)
{
   __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
   vector = _mm512_permutexvar_epi8(SHUFFLE_REMOVE[index], vector);
   _mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}

static inline void update_tags_512(vqf_block<16>* restrict block, uint8_t index, uint16_t tag)
{
   block->tags[27] = tag;  // add tag at the end

   __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
   vector = _mm512_permutexvar_epi16(SHUFFLE16[index], vector);
   _mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}

static inline void remove_tags_512(vqf_block<16>* restrict block, uint8_t index)
{
   __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
   vector = _mm512_permutexvar_epi16(SHUFFLE_REMOVE16[index], vector);
   _mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}

#else  // __AVX512BW__

static inline void update_tags_512(vqf_block<8>* restrict block, uint8_t index, uint8_t tag)
{
   index -= 16;
   memmove(&block->tags[index + 1], &block->tags[index],
           sizeof(block->tags) / sizeof(block->tags[0]) - index - 1);
   block->tags[index] = tag;
}

static inline void remove_tags_512(vqf_block<8>* restrict block, uint8_t index)
{
   index -= 16;
   memmove(&block->tags[index], &block->tags[index + 1],
           sizeof(block->tags) / sizeof(block->tags[0]) - index);
}

static inline void update_tags_512(vqf_block<16>* restrict block, uint8_t index, uint16_t tag)
{
   index -= 4;
   memmove(&block->tags[index + 1], &block->tags[index],
           (sizeof(block->tags) / sizeof(block->tags[0]) - index - 1) * 2);
   block->tags[index] = tag;
}

static inline void remove_tags_512(vqf_block<16>* restrict block, uint8_t index)
{
   index -= 4;
   memmove(&block->tags[index], &block->tags[index + 1],
           (sizeof(block->tags) / sizeof(block->tags[0]) - index) * 2);
}

#endif  // __AVX512BW__

#if 0
// Shuffle using AVX2 vector instruction. It turns out memmove is faster compared to AVX2.
inline __m256i cross_lane_shuffle(const __m256i & value, const __m256i &
      shuffle) 
{ 
   return _mm256_or_si256(_mm256_shuffle_epi8(value, _mm256_add_epi8(shuffle,
               K[0])), 
         _mm256_shuffle_epi8(_mm256_permute4x64_epi64(value, 0x4E),
            _mm256_add_epi8(shuffle, K[1]))); 
}

#define SHUFFLE_SIZE 32
void shuffle_256(uint8_t * restrict source, __m256i shuffle) {
   __m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source)); 

   vector = cross_lane_shuffle(vector, shuffle); 
   _mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector); 
} 

static inline void update_tags_256(uint8_t * restrict block, uint8_t index,
      uint8_t tag) {
   index = index + sizeof(__uint128_t);	// offset index based on md field.
   block[63] = tag;	// add tag at the end
   shuffle_256(block + SHUFFLE_SIZE, RM[index]); // right block shuffle
   if (index < SHUFFLE_SIZE) {		// if index lies in the left block
      std::swap(block[31], block[32]);	// move tag to the end of left block
      shuffle_256(block, LM[index]);	// shuffle left block
   }
}
#endif

template <int TAG_BITS>
struct vqf_md;

template <>
struct vqf_md<8> {
   static inline void update_md(uint64_t* md, uint8_t index)
   {
      uint64_t carry = (md[0] >> 63) & carry_pdep_table[index];
      md[1] = _pdep_u64(md[1], high_order_pdep_table[index]) | carry;
      md[0] = _pdep_u64(md[0], low_order_pdep_table[index]);
   }

   static inline void remove_md(uint64_t* md, uint8_t index)
   {
      uint64_t carry = (md[1] & carry_pdep_table[index]) << 63;
      md[1] = _pext_u64(md[1], high_order_pdep_table[index]) | (1ULL << 63);
      md[0] = _pext_u64(md[0], low_order_pdep_table[index]) | carry;
   }

   // number of 0s in the metadata is the number of tags.
   static inline uint64_t get_block_free_space(uint64_t* vector)
   {
      uint64_t lower_word = vector[0];
      uint64_t higher_word = vector[1];
      return word_rank(lower_word) + word_rank(higher_word);
   }
};

template <>
struct vqf_md<16> {
   static inline void update_md(uint64_t* md, uint8_t index)
   {
      *md = _pdep_u64(*md, low_order_pdep_table[index]);
   }

   static inline void remove_md(uint64_t* md, uint8_t index)
   {
      *md = _pext_u64(*md, low_order_pdep_table[index]) | (1ULL << 63);
   }

   // number of 0s in the metadata is the number of tags.
   static inline uint64_t get_block_free_space(uint64_t vector)
   {
      return word_rank(vector);
   }
};

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
template <int TAG_BITS>
uint64_t vqf_filter_size(const vqf_filter<TAG_BITS>* restrict filter)
{
   return sizeof(*filter) + filter->metadata.total_size_in_bytes;
}

template uint64_t vqf_filter_size<8>(const vqf_filter<8>* restrict filter);
template uint64_t vqf_filter_size<16>(const vqf_filter<16>* restrict filter);

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
template <int TAG_BITS>
void vqf_free(vqf_filter<TAG_BITS>** restrict filter)
{
   free(*filter);
   *filter = nullptr;
}

template void vqf_free<8>(vqf_filter<8>** restrict filter);
template void vqf_free<16>(vqf_filter<16>** restrict filter);

//=#=#==#==#===============+=+=+=+=++=++++++++++++++-++-+--+-+----+---------------

template <int TAG_BITS>
struct vqf_filter_init_params {
   uint64_t total_blocks;
   uint64_t total_size_in_bytes;

   explicit vqf_filter_init_params(uint64_t nslots) noexcept
       : total_blocks{(nslots + vqf_constants<TAG_BITS>::QUQU_SLOTS_PER_BLOCK) /
                      vqf_constants<TAG_BITS>::QUQU_SLOTS_PER_BLOCK}

       , total_size_in_bytes{sizeof(vqf_block<TAG_BITS>) * total_blocks}
   {
   }
};

void vqf_filter_init_blocks(vqf_filter<8>* filter, uint64_t total_blocks);
void vqf_filter_init_blocks(vqf_filter<16>* filter, uint64_t total_blocks);

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
uint64_t vqf_nslots_for_size(int tag_bits, uint64_t target_byte_size)
{
   uint64_t z = target_byte_size;
   uint64_t b, q, h;

   switch (tag_bits) {
   case 8:
      h = sizeof(vqf_filter<8>);
      b = sizeof(vqf_block<8>);
      q = vqf_constants<8>::QUQU_SLOTS_PER_BLOCK;
      break;

   case 16:
      h = sizeof(vqf_filter<16>);
      b = sizeof(vqf_block<16>);
      q = vqf_constants<16>::QUQU_SLOTS_PER_BLOCK;
      break;

   default:
      std::cerr << "ILLEGAL TAG_BITS VALUE: " << tag_bits << std::endl;
      std::terminate();
   }

   uint64_t n = q * ((z - h) - 1) / b;
   uint64_t check_z = ((n + q) / q) * b + h;

   if (z < check_z) {
      n -= q;
      check_z = ((n + q) / q) * b + h;
      if (z < check_z) {
         std::cerr << "Check the math! " << z << " vs " << check_z << std::endl;
         std::terminate();
      }
   }

   return n;
}

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
// Create n/log(n) blocks of log(n) slots.
// log(n) is 51 given a cache line size.
// n/51 blocks.
//
template <int TAG_BITS>
vqf_filter<TAG_BITS>* vqf_init(uint64_t nslots)
{
   vqf_filter<TAG_BITS>* filter;

   vqf_filter_init_params<TAG_BITS> params{nslots};

   filter = (vqf_filter<TAG_BITS>*)malloc(sizeof(*filter) + params.total_size_in_bytes);
   assert(filter);

   return vqf_init_in_place(filter, nslots);
}

// Instantiate for supported values of TAG_BITS.
//
template vqf_filter<8>* vqf_init<8>(uint64_t nslots);
template vqf_filter<16>* vqf_init<16>(uint64_t nslots);

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
template <int TAG_BITS>
vqf_filter<TAG_BITS>* vqf_init_in_place(vqf_filter<TAG_BITS>* restrict filter, uint64_t nslots)
{
   vqf_filter_init_params<TAG_BITS> params{nslots};

   filter->metadata.total_size_in_bytes = params.total_size_in_bytes;
   filter->metadata.nslots = params.total_blocks * vqf_constants<TAG_BITS>::QUQU_SLOTS_PER_BLOCK;
   filter->metadata.key_remainder_bits = TAG_BITS;
   filter->metadata.range = params.total_blocks * vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;
   filter->metadata.nblocks = params.total_blocks;
   filter->metadata.nelts = 0;

   // memset to 1
   //
   vqf_filter_init_blocks(filter, params.total_blocks);

   return filter;
}

//+++++++++++-+-+--+----- --- -- -  -  -   -
//
void vqf_filter_init_blocks(vqf_filter<8>* filter, uint64_t total_blocks)
{
   for (uint64_t i = 0; i < total_blocks; i++) {
      filter->blocks[i].md[0] = UINT64_MAX;
      filter->blocks[i].md[1] = UINT64_MAX;
      // reset the most significant bit of metadata for locking.
      filter->blocks[i].md[1] = filter->blocks[i].md[1] & ~(1ULL << 63);
   }
}

//+++++++++++-+-+--+----- --- -- -  -  -   -
//
void vqf_filter_init_blocks(vqf_filter<16>* filter, uint64_t total_blocks)
{
   for (uint64_t i = 0; i < total_blocks; i++) {
      filter->blocks[i].md = UINT64_MAX;
      filter->blocks[i].md = filter->blocks[i].md & ~(1ULL << 63);
   }
}

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
uint64_t alt_index(uint64_t index, uint64_t tag, uint64_t range)
{
   return (uint64_t)(range - index + (tag * 0x5bd1e995)) % range;
}

//=#=#==#==#===============+=+=+=+=++=++++++++++++++-++-+--+-+----+---------------

template <int TAG_BITS>
struct vqf_insert_impl;

template <>
struct vqf_insert_impl<8> {
   static constexpr int TAG_BITS = 8;

   static uint64_t* init_block_md(vqf_block<TAG_BITS>* restrict blocks, uint64_t block_index)
   {
      return blocks[block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK].md;
   }

   static uint64_t init_block_free(uint64_t* block_md)
   {
      return vqf_md<TAG_BITS>::get_block_free_space(block_md);
   }

   static uint64_t* init_alt_block_md(vqf_block<TAG_BITS>* restrict blocks, uint64_t alt_block_index)
   {
      return blocks[alt_block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK].md;
   }

   static uint64_t init_alt_block_free(uint64_t* alt_block_md)
   {
      return vqf_md<TAG_BITS>::get_block_free_space(alt_block_md);
   }

   static uint64_t init_slot_index(uint64_t* block_md, uint64_t offset)
   {
      return select_128(block_md, offset);
   }

   static uint64_t init_select_index(uint64_t slot_index, uint64_t offset)
   {
      return slot_index + offset - sizeof(__uint128_t);
   }
};

//+++++++++++-+-+--+----- --- -- -  -  -   -

template <>
struct vqf_insert_impl<16> {
   static constexpr int TAG_BITS = 16;

   static uint64_t* init_block_md(vqf_block<TAG_BITS>* restrict blocks, uint64_t block_index)
   {
      return &blocks[block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK].md;
   }

   static uint64_t init_block_free(uint64_t* block_md)
   {
      return vqf_md<TAG_BITS>::get_block_free_space(*block_md);
   }

   static uint64_t* init_alt_block_md(vqf_block<TAG_BITS>* restrict blocks, uint64_t alt_block_index)
   {
      return &blocks[alt_block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK].md;
   }

   static uint64_t init_alt_block_free(uint64_t* alt_block_md)
   {
      return vqf_md<TAG_BITS>::get_block_free_space(*alt_block_md);
   }

   static uint64_t init_slot_index(uint64_t* block_md, uint64_t offset)
   {
      return select_64(*block_md, offset);
   }

   static uint64_t init_select_index(uint64_t slot_index, uint64_t offset)
   {
      return slot_index + offset - (sizeof(uint64_t) / 2);
   }
};

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
// If the item goes in the i'th slot (starting from 0) in the block then
// find the i'th 0 in the metadata, insert a 1 after that and shift the rest
// by 1 bit.
// Insert the new tag at the end of its run and shift the rest by 1 slot.
//
template <int TAG_BITS>
bool vqf_insert(vqf_filter<TAG_BITS>* restrict filter, uint64_t hash)
{
   vqf_metadata* restrict metadata = &filter->metadata;
   vqf_block<TAG_BITS>* restrict blocks = filter->blocks;
   uint64_t key_remainder_bits = metadata->key_remainder_bits;
   uint64_t range = metadata->range;

   uint64_t block_index = hash % range;
   lock(blocks[block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);

   uint64_t* block_md = vqf_insert_impl<TAG_BITS>::init_block_md(blocks, block_index);
   uint64_t block_free = vqf_insert_impl<TAG_BITS>::init_block_free(block_md);
   uint64_t tag = (hash >> 32) & vqf_constants<TAG_BITS>::TAG_MASK;
   tag += (tag == 0);
   uint64_t alt_block_index = alt_index(block_index, tag, range);

   __builtin_prefetch(&blocks[alt_block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);

   if (block_free < vqf_constants<TAG_BITS>::QUQU_CHECK_ALT &&
       block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK !=
           alt_block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK) {
      unlock(blocks[block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
      lock_blocks(filter, block_index, alt_block_index);

      uint64_t* alt_block_md = vqf_insert_impl<TAG_BITS>::init_alt_block_md(blocks, alt_block_index);
      uint64_t alt_block_free = vqf_insert_impl<TAG_BITS>::init_alt_block_free(alt_block_md);

      // pick the least loaded block
      if (alt_block_free > block_free) {
         unlock(blocks[block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
         block_index = alt_block_index;
         block_md = alt_block_md;
      } else if (block_free == vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK) {
         unlock_blocks(filter, block_index, alt_block_index);
         return false;

      } else {
         unlock(blocks[alt_block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
      }
   }

   uint64_t index = block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;

   uint64_t slot_index = vqf_insert_impl<TAG_BITS>::init_slot_index(block_md, offset);
   uint64_t select_index = vqf_insert_impl<TAG_BITS>::init_select_index(slot_index, offset);

   update_tags_512(&blocks[index], slot_index, tag);
   vqf_md<TAG_BITS>::update_md(block_md, select_index);

   unlock(blocks[block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);
   return true;
}

template bool vqf_insert<8>(vqf_filter<8>* restrict filter, uint64_t hash);
template bool vqf_insert<16>(vqf_filter<16>* restrict filter, uint64_t hash);

//=#=#==#==#===============+=+=+=+=++=++++++++++++++-++-+--+-+----+---------------

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
static inline bool remove_tags(vqf_filter<8>* restrict filter, uint64_t tag, uint64_t block_index)
{
   static constexpr int TAG_BITS = 8;

   uint64_t index = block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;

#ifdef __AVX512BW__
   __m512i bcast = _mm512_set1_epi8(tag);
   __m512i block = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index]));
   volatile __mmask64 result = _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
#else
   __m256i bcast = _mm256_set1_epi8(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index]));
   __m256i result1t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);

   block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index] + 32));
   __m256i result2t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);

   uint64_t result = (uint64_t)result2 << 32 | (uint64_t)result1;
#endif

   if (result == 0) {
      // no matching tags, can bail
      return false;
   }

   uint64_t start =
       offset != 0 ? lookup_128(filter->blocks[index].md, offset - 1) : one[0] << 2 * sizeof(uint64_t);
   uint64_t end = lookup_128(filter->blocks[index].md, offset);
   uint64_t mask = end - start;

   uint64_t check_indexes = mask & result;
   if (check_indexes != 0) {  // remove the first available tag
      vqf_block<TAG_BITS>* restrict blocks = filter->blocks;
      uint64_t remove_index = __builtin_ctzll(check_indexes);
      remove_tags_512(&blocks[index], remove_index);
      remove_index = remove_index + offset - sizeof(__uint128_t);
      uint64_t* block_md = blocks[block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK].md;
      vqf_md<TAG_BITS>::remove_md(block_md, remove_index);
      return true;
   } else {
      return false;
   }
}

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
static inline bool remove_tags(vqf_filter<16>* restrict filter, uint64_t tag, uint64_t block_index)
{
   static constexpr int TAG_BITS = 16;

   uint64_t index = block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;

#ifdef __AVX512BW__
   __m512i bcast = _mm512_set1_epi16(tag);
   __m512i block = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index]));
   volatile __mmask64 result = _mm512_cmp_epi16_mask(bcast, block, _MM_CMPINT_EQ);
#else
   uint64_t alt_mask = 0x55555555;
   __m256i bcast = _mm256_set1_epi16(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index]));
   __m256i result1t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);
   result1 = _pext_u32(result1, alt_mask);

   block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index] + 32));
   __m256i result2t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);
   result2 = _pext_u32(result2, alt_mask);

   uint64_t result = (uint64_t)result2 << 16 | (uint64_t)result1;
#endif

   if (result == 0) {
      // no matching tags, can bail
      return false;
   }

   uint64_t start =
       offset != 0 ? lookup_64(filter->blocks[index].md, offset - 1) : one[0] << (sizeof(uint64_t) / 2);
   uint64_t end = lookup_64(filter->blocks[index].md, offset);
   uint64_t mask = end - start;

   uint64_t check_indexes = mask & result;
   if (check_indexes != 0) {  // remove the first available tag
      vqf_block<TAG_BITS>* restrict blocks = filter->blocks;
      uint64_t remove_index = __builtin_ctzll(check_indexes);
      remove_tags_512(&blocks[index], remove_index);
      remove_index = remove_index + offset - sizeof(uint64_t);
      uint64_t* block_md = &blocks[block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK].md;
      vqf_md<TAG_BITS>::remove_md(block_md, remove_index);
      return true;
   } else {
      return false;
   }
}

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
template <int TAG_BITS>
bool vqf_remove(vqf_filter<TAG_BITS>* restrict filter, uint64_t hash)
{
   vqf_metadata* restrict metadata = &filter->metadata;
   uint64_t key_remainder_bits = metadata->key_remainder_bits;
   uint64_t range = metadata->range;

   uint64_t block_index = hash % range;
   uint64_t tag = (hash >> 32) & vqf_constants<TAG_BITS>::TAG_MASK;
   tag += (tag == 0);
   uint64_t alt_block_index = alt_index(block_index, tag, range);

   __builtin_prefetch(&filter->blocks[alt_block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);

   return remove_tags(filter, tag, block_index) || remove_tags(filter, tag, alt_block_index);
}

template bool vqf_remove<8>(vqf_filter<8>* restrict filter, uint64_t hash);
template bool vqf_remove<16>(vqf_filter<16>* restrict filter, uint64_t hash);

//=#=#==#==#===============+=+=+=+=++=++++++++++++++-++-+--+-+----+---------------

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
static inline bool check_tags(const vqf_filter<8>* restrict filter, uint64_t tag, uint64_t block_index)
{
   static constexpr int TAG_BITS = 8;

   uint64_t index = block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;

#ifdef __AVX512BW__
   __m512i bcast = _mm512_set1_epi8(tag);
   __m512i block = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&filter->blocks[index]));
   volatile __mmask64 result = _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
#else
   __m256i bcast = _mm256_set1_epi8(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&filter->blocks[index]));
   __m256i result1t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);

   block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((const uint8_t*)&filter->blocks[index] + 32));
   __m256i result2t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);

   uint64_t result = (uint64_t)result2 << 32 | (uint64_t)result1;
#endif

   if (result == 0) {
      // no matching tags, can bail
      return false;
   }

   uint64_t start =
       offset != 0 ? lookup_128(filter->blocks[index].md, offset - 1) : one[0] << 2 * sizeof(uint64_t);
   uint64_t end = lookup_128(filter->blocks[index].md, offset);
   uint64_t mask = end - start;
   return (mask & result) != 0;
}

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
static inline bool check_tags(const vqf_filter<16>* restrict filter, uint64_t tag, uint64_t block_index)
{
   static constexpr int TAG_BITS = 16;

   uint64_t index = block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK;

#ifdef __AVX512BW__
   __m512i bcast = _mm512_set1_epi16(tag);
   __m512i block = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&filter->blocks[index]));
   volatile __mmask64 result = _mm512_cmp_epi16_mask(bcast, block, _MM_CMPINT_EQ);
#else
   uint64_t alt_mask = 0x55555555;
   __m256i bcast = _mm256_set1_epi16(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&filter->blocks[index]));
   __m256i result1t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);
   result1 = _pext_u32(result1, alt_mask);

   block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((const uint8_t*)&filter->blocks[index] + 32));
   __m256i result2t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);
   result2 = _pext_u32(result2, alt_mask);

   uint64_t result = (uint64_t)result2 << 16 | (uint64_t)result1;
#endif

   if (result == 0) {
      // no matching tags, can bail
      return false;
   }

   uint64_t start =
       offset != 0 ? lookup_64(filter->blocks[index].md, offset - 1) : one[0] << (sizeof(uint64_t) / 2);
   uint64_t end = lookup_64(filter->blocks[index].md, offset);
   uint64_t mask = end - start;
   return (mask & result) != 0;
}

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
// If the item goes in the i'th slot (starting from 0) in the block then
// select(i) - i is the slot index for the end of the run.
//
template <int TAG_BITS>
bool vqf_is_present(const vqf_filter<TAG_BITS>* restrict filter, uint64_t hash)
{
   const vqf_metadata* restrict metadata = &filter->metadata;
   uint64_t key_remainder_bits = metadata->key_remainder_bits;
   uint64_t range = metadata->range;

   uint64_t block_index = hash % range;
   uint64_t tag = (hash >> 32) & vqf_constants<TAG_BITS>::TAG_MASK;
   tag += (tag == 0);

   uint64_t alt_block_index = alt_index(block_index, tag, range);

   __builtin_prefetch(&filter->blocks[alt_block_index / vqf_constants<TAG_BITS>::QUQU_BUCKETS_PER_BLOCK]);

   return check_tags(filter, tag, block_index) || check_tags(filter, tag, alt_block_index);
}

template bool vqf_is_present<8>(const vqf_filter<8>* restrict filter, uint64_t hash);
template bool vqf_is_present<16>(const vqf_filter<16>* restrict filter, uint64_t hash);
