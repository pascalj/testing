#pragma once

#include <alpaka/alpaka.hpp>
#include <mephisto/internal/iterator.hpp>

namespace mephisto {
namespace execution {

template <typename T, uint64_t size>
struct cheapArray
{
    T data[size];

    //-----------------------------------------------------------------------------
    //! Access operator.
    //!
    //! \param index The index of the element to be accessed.
    //!
    //! Returns the requested element per reference.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T &operator[](uint64_t index)
    {
        return data[index];
    }

    //-----------------------------------------------------------------------------
    //! Access operator.
    //!
    //! \param index The index of the element to be accessed.
    //!
    //! Returns the requested element per constant reference.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const T &operator[](uint64_t index) const
    {
        return data[index];
    }
};

//#############################################################################
//! A reduction kernel.
//!
//! \tparam TBlockSize The block size.
//! \tparam T The data type.
//! \tparam TFunc The Functor type for the reduction function.
template <typename TFunc>
struct ReduceKernel
{
    ALPAKA_NO_HOST_ACC_WARNING

    TFunc func;

    ReduceKernel(TFunc func)
      : func(func)
    {
    }

    static constexpr std::size_t TBlockSize = 256;

    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment.
    //! \tparam TElem The element type.
    //! \tparam TIdx The index type.
    //!
    //! \param acc The accelerator object.
    //! \param source The source memory.
    //! \param destination The destination memory.
    //! \param n The problem size.
    //! \param func The reduction function.

    template <typename TAcc, typename TElem, typename TRes, typename TIdx>
    ALPAKA_FN_ACC void operator()(
        TAcc const &      acc,
        const std::size_t block_index,
        TElem *const      source,
        TRes *            destination,
        TIdx const &      n) const
    {
      /* std::cout << "reduce kernel" << std::endl; */
      auto &sdata(
          alpaka::block::shared::st::
              allocVar<cheapArray<TRes, TBlockSize>, __COUNTER__>(acc));

      const uint32_t blockIndex(static_cast<uint32_t>(
          alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]));
      const uint32_t threadIndex(static_cast<uint32_t>(
          alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]));
      const uint32_t gridDimension(static_cast<uint32_t>(
          alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]));

      // equivalent to blockIndex * TBlockSize + threadIndex
      const uint32_t linearizedIndex(static_cast<uint32_t>(
          alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]));

      IteratorCpu<TAcc, TElem> begin(
          acc, source, linearizedIndex, gridDimension * TBlockSize, n);
      auto it = begin;

      // doing the remaining blocks
      while (it < it.end()) func(blockIndex, 0, sdata.data, &(*it++));

      alpaka::block::sync::syncBlockThreads(acc);

      // --------
      // Level 2: block + warp reduce, reading from shared memory
      // --------

      printf("sdata: ");
      for(size_t i = 0; i < TBlockSize; i++) printf("%d ", sdata[i]);
      ALPAKA_UNROLL()
      for (uint32_t currentBlockSize = TBlockSize,
                    currentBlockSizeUp =
                        (TBlockSize + 1) / 2;  // ceil(TBlockSize/2.0)
           currentBlockSize > 1;
           currentBlockSize            = currentBlockSize / 2,
                    currentBlockSizeUp = (currentBlockSize + 1) /
                                         2)  // ceil(currentBlockSize/2.0)
      {
        bool cond =
            threadIndex < currentBlockSizeUp  // only first half of block
                                              // is working
            && (threadIndex + currentBlockSizeUp) <
                   TBlockSize  // index for second half must be in bounds
            && (blockIndex * TBlockSize + threadIndex + currentBlockSizeUp) <
                   n &&
            threadIndex <
                n;  // if elem in second half has been initialized before

        if (cond) {
          sdata[threadIndex] = sdata[threadIndex] + sdata[threadIndex + currentBlockSizeUp];
        }

        alpaka::block::sync::syncBlockThreads(acc);
      }

      // store block result to gmem
      if (threadIndex == 0 && threadIndex < n) {
        printf("\ndestination[%zu]: %d\n", block_index, sdata[0]);
        destination[block_index] = sdata[0];
      }
    }
};

template <typename F>
class UniversalKernel {
  F _func;
public:

  UniversalKernel(F func) : _func(func) {}

  ALPAKA_NO_HOST_ACC_WARNING
  template <typename TAcc, typename TElem, typename TRes, typename TIdx>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &      acc,
      const std::size_t block_index,
      TElem *const      A,
      TRes *            res,
      TIdx const &      numElements) const -> void
  {
    /* const uint32_t blockIndex(static_cast<uint32_t>( */
    /*     alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0])); */
    /* const uint32_t threadIndex(static_cast<uint32_t>( */
    /*     alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0])); */
    /* const uint32_t gridDimension(static_cast<uint32_t>( */
    /*     alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0])); */
    /* const uint32_t threadElementExtent(static_cast<uint32_t>( */
    /*     alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc).prod())); */

    /* const uint32_t i(static_cast<uint32_t>( */
    /*     alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0])); */

    /* TIdx const threadFirstElemIdx(i * threadElementExtent); */

    /* if (threadFirstElemIdx < numElements) { */
    /*   TIdx const threadLastElemIdx(threadFirstElemIdx + threadElementExtent); */
    /*   TIdx const threadLastElemIdxClipped( */
    /*       (numElements > threadLastElemIdx) ? threadLastElemIdx */
    /*                                         : numElements); */
    _func(acc, block_index, res, A);
  }
};
}  // namespace execution
}  // namespace mephisto
