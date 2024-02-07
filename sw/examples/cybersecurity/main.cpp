#include <iostream>
#include <string>

#include "cProcess.hpp"

#define ASYNC
// #define TRANSFER

using namespace std;
using namespace fpga;

int main(int argc, char *argv[]) {

  std::unique_ptr<cProcess> cproc;

  const uint64_t max_size = 1ULL << 17ULL;
  uint32_t n_pages = ((max_size + pageSize - 1) / pageSize);

  // Obtain resources
  cproc = std::make_unique<cProcess>(0, getpid());
  int8_t *hMem =
      static_cast<int8_t *>(cproc->getMem({CoyoteAlloc::REG_4K, n_pages}));
  if (!hMem) {
    std::cout << "Unable to allocate mem" << std::endl;
    return 1;
  }

  memset(hMem, 0, max_size);

  const auto nbInputs = 1ULL;
  const auto inputSize = 16ULL;

  const auto inferenceDataSize2D = nbInputs * inputSize;
  for (uint64_t i = 0; i < nbInputs; ++i) {
    for (uint64_t j = 0; j < inputSize; ++j) {
      hMem[i * inputSize + j] = -1;
    }
  }

  std::cout << "Setting the number of batches to: " << nbInputs << std::endl;

  uint64_t FINN_CTRL_OFFSET = 0x0;

  cproc->setCSR(nbInputs, (FINN_CTRL_OFFSET + 0x10) >> 3);
  cproc->setCSR(0x8110, (FINN_CTRL_OFFSET + 0x18) >> 3);
  cproc->setCSR(1, (FINN_CTRL_OFFSET + 0x0) >> 3);
  while (!(cproc->getCSR((FINN_CTRL_OFFSET + 0x0) >> 3) & 2))
    ;

  std::cout << "Before kernel, data in write region is:" << std::endl;
  for (uint64_t i = inferenceDataSize2D; i < inferenceDataSize2D + nbInputs;
       ++i) {
    std::cout << "hMem[" << i << "] = " << std::to_string(hMem[i]) << std::endl;
  }

  csInvoke invokeRead = {
      .oper = CoyoteOper::READ, .addr = &hMem[0], .len = inferenceDataSize2D};
  csInvoke asyncRead = {.oper = CoyoteOper::READ,
                        .addr = &hMem[0],
                        .len = inferenceDataSize2D,
                        .clr_stat = false,
                        .poll = false,
                        .dest = 0,
                        .stream = true};

  csInvoke invokeWrite = {.oper = CoyoteOper::WRITE,
                          .addr = &hMem[inferenceDataSize2D],
                          .len = nbInputs};

  csInvokeAll invokeTransfer = {.oper = CoyoteOper::TRANSFER,
                                .src_addr = &hMem[0],
                                .dst_addr = &hMem[inferenceDataSize2D],
                                .src_len = inferenceDataSize2D,
                                .dst_len = nbInputs,
                                .clr_stat = false,
                                .poll = false,
                                .dest = 0,
                                .stream = true};
  constexpr auto NB_EXPERIMENTS = 100;
  std::ofstream outfile;
  outfile.open("results.txt"); // append instead of overwrite
  for (int j = 0; j < NB_EXPERIMENTS; ++j) {
    const auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NB_EXPERIMENTS; ++i) {

#ifdef TRANSFER
      cproc->invoke(invokeTransfer);
      while (!cproc->checkCompleted(CoyoteOper::TRANSFER))
        ;
#else
#ifdef ASYNC
      cproc->invoke(asyncRead);
      cproc->invoke(invokeWrite);
#else
      cproc->invoke(invokeRead);
      cproc->invoke(invokeWrite);
#endif
#endif
    }

    const auto end = std::chrono::high_resolution_clock::now();
    cproc->clearCompleted();
    const auto diffNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count();
    outfile << diffNs << '\n';
    // std::cout << "Diff in nanosec is: " << diffNs << std::endl;
  }

  std::cout << "After kernel, data in write region is:" << std::endl;
  for (uint64_t i = inferenceDataSize2D; i < inferenceDataSize2D + nbInputs;
       ++i) {
    std::cout << "hMem[" << i << "] = " << std::to_string(hMem[i]) << std::endl;
  }

  return EXIT_SUCCESS;
}

