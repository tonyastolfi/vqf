#include <vqf/vqf.hpp>
//
#include <vqf/vqf.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <batteries/metrics/metric_collectors.hpp>

#include <batteries/finally.hpp>
#include <batteries/int_types.hpp>
#include <batteries/stream_util.hpp>

#include <functional>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace {

using namespace batt::int_types;

using batt::Every2ToTheConst;
using batt::LatencyMetric;
using batt::LatencyTimer;
using batt::StatsMetric;

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
template <usize kLength = 7, typename Rng>
std::string pick_key(Rng& rng, std::integral_constant<usize, kLength> = {})
{
   std::uniform_int_distribution<char> pick_char{'a', 'z'};
   std::array<char, kLength> chars;

   for (char& ch : chars) {
      ch = pick_char(rng);
   }

   return std::string(chars.data(), chars.size());
}

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
template <int TAG_BITS>
void run_filter_test()
{
   LatencyMetric query_latency;

   const u32 random_seed = 815763;
   const usize n_keys = 15000;
   const usize n_slots = 24000;
   const usize n_latency_queries = 100000;

   vqf_filter<TAG_BITS>* filter = vqf_init<TAG_BITS>(n_slots);

   ASSERT_NE(filter, nullptr);
   EXPECT_GT(vqf_filter_size(filter), n_slots);

   auto on_scope_exit = batt::finally([&] {
      vqf_free(&filter);
      EXPECT_EQ(filter, nullptr);
   });

   //+++++++++++-+-+--+----- --- -- -  -  -   -

   std::default_random_engine rng{random_seed};
   std::vector<std::string> key_vec;

   for (usize i = 0; i < n_keys; ++i) {
      std::string key = pick_key(rng);
      key_vec.emplace_back(key);
      u64 hash_val = std::hash<std::string>{}(key);

      ASSERT_TRUE(vqf_insert(filter, hash_val));
   }

   std::unordered_set<std::string> key_set(key_vec.begin(), key_vec.end());

   for (const std::string& key : key_vec) {
      u64 hash_val = std::hash<std::string>{}(key);

      ASSERT_TRUE(vqf_is_present(filter, hash_val));
   }

   //+++++++++++-+-+--+----- --- -- -  -  -   -

   {
      std::vector<std::string> query_keys;
      for (usize i = 0; i < n_latency_queries; ++i) {
         query_keys.emplace_back(pick_key(rng));
      }

      usize is_present_count = 0;

      LatencyTimer timer{query_latency, query_keys.size()};

      for (const std::string& key : query_keys) {
         const u64 hash_val = std::hash<std::string>{}(key);
         const bool filter_result = vqf_is_present(filter, hash_val);
         is_present_count += filter_result;
      }

      timer.stop();

      std::cerr << BATT_INSPECT(is_present_count) << std::endl;
   }

   //+++++++++++-+-+--+----- --- -- -  -  -   -

   double positive_queries = 0;
   double negative_queries = 0;
   double false_positives = 0;

   for (usize j = 0; j < key_vec.size() * 1000 || false_positives < 10; ++j) {
      std::string query_key = pick_key(rng);
      const u64 hash_val = std::hash<std::string>{}(query_key);
      const bool filter_result = vqf_is_present(filter, hash_val);
      const bool key_in_set = key_set.count(query_key);

      if (key_in_set) {
         positive_queries += 1;
      } else {
         negative_queries += 1;
      }

      if (filter_result) {
         if (!key_in_set) {
            false_positives += 1;
         }
      } else {
         ASSERT_FALSE(key_in_set);
      }
   }

   std::cerr << "bits/key=" << TAG_BITS << " size=" << vqf_filter_size(filter)
             << " ε=" << (false_positives / negative_queries) << " keys=" << key_vec.size()
             << BATT_INSPECT(query_latency) << std::endl;
}

//==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
//
TEST(VqfTest, Test)
{
   run_filter_test<8>();
   run_filter_test<16>();
}

}  // namespace
