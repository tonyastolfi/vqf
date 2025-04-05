/*
 * ============================================================================
 *
 *       Filename:  vqf_wrapper.h
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */

#ifndef VQF_WRAPPER_H
#define VQF_WRAPPER_H

#include "vqf_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

vqf_filter* q_filter;

inline int q_init(uint64_t nbits)
{
	uint64_t nslots = (1ULL << nbits);
	q_filter = vqf_init(nslots);
	return 0;
}

inline int q_insert(__uint128_t val)
{
	if (!vqf_insert(q_filter, val))
		return 0;
	return 1;
}

inline int q_lookup(__uint128_t val)
{
	if (!vqf_is_present(q_filter, val))
		return 0;
	return 1;
}

inline int q_remove(__uint128_t val)
{
	if (!vqf_remove(q_filter, val))
		return 0;
	return 1;
}

inline __uint128_t q_range()
{
	//return q_filter->metadata.range;
	return UINT64_MAX;
}

inline int q_destroy()
{
	return 0;
}

#ifdef __cplusplus
}
#endif

#endif
