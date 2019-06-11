#include "gtest/gtest.h"
#include "transform_reduce_test.h"
#include <libdash.h>
#include <mephisto/algorithm/for_each>
#include <mephisto/execution>
#include <mephisto/array>
#include <mephisto/entity>
#include <patterns/local_pattern.h>
#include <alpaka/alpaka.hpp>

TEST_F(TransformReduceTest, MinElement) {
  auto const Dim = 2;
  using ViewT    = typename dash::Array<Data>::local_type;
  using SizeT    = ArrayT::size_type;
  using EntityT =
      mephisto::Entity<Dim, std::size_t, alpaka::acc::AccCpuSerial>;
  using Queue   = alpaka::queue::QueueCpuSync;
  using Context = mephisto::execution::AlpakaExecutionContext<EntityT, Queue>;
  using BasePattern = dash::BlockPattern<Dim>;
  using PatternT    = patterns::BalancedLocalPattern<BasePattern, EntityT>;
  using ArrayT      = dash::Array<Data, dash::default_index_t, PatternT>;

  BasePattern base{5, 5};
  PatternT pattern{base};
  ArrayT   arr{pattern};
  dash::fill(arr.begin(), arr.end(), 42);

  // Setup of the executor:

  // Context consists of the host, the accelerator and the stream
  Context ctx;

  // The executor is the one actually doing the computation
  mephisto::execution::AlpakaExecutor<Context> executor{ctx};

  // The policy is used to relax guarantees.
  auto policy = mephisto::execution::make_parallel_policy(executor);

  // set the coordinates using an Alpaka policy
  dash::transform_reduce(
      policy,
      arr.begin(),
      arr.end(),
      Data{0},
      [](const Data sum, const Data i) { return sum + i; },
      [](const auto i) { return i + 13; });

  // Check the written coordinates using the standard for_each_with_index
  dash::for_each(arr.begin(), arr.end(), [](const Data &d) {
    std::cout << "Result :" << d << std::endl;
  });
}


int main(int argc, char **argv) {
  meph_argc = argc;
  meph_argv = argv;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

