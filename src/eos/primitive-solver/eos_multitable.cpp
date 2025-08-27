//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_multitable.cpp
//  \brief Implementation of EOSMultiTable

#include <math.h>

#include <cassert>
#include <cstdio>
#include <limits>
#include <iostream>
#include <cstddef>
#include <string>

#include <Kokkos_Core.hpp>

#include "eos_multitable.hpp"
#include "athena.hpp"
#include "utils/tr_table.hpp"
#include "logs.hpp"

namespace Primitive {

template<typename LogPolicy>
void EOSMultiTable<LogPolicy>::ReadTableFromFile(std::string fname) {
  if (initialised==false) {

    std::ifstream file;
    try {
      file.open(fname.c_str(), std::ifstream::in);
    } catch (std::ifstream::failure& e) {
      return;
    }

    if (!file.is_open()) {
      return;
    }

    // TODO LOOP Read header file
    char buffer[256];
    file.getline(buffer, 256);
    std::string line = std::string(buffer);

    // TODO LOOP read tables

    initialised = true;
  }

  return;
}

template<typename LogPolicy>
bool EOSMultiTable<LogPolicy>::Read2DTableFromFile(std::string fname) {
  bool success = false;
  //TODO
  return success;
}


template<typename LogPolicy>
bool EOSMultiTable<LogPolicy>::Read3DTableFromFile(std::string fname) {
  bool success = false;
  //TODO
  return success;
}

template class EOSMultiTable<NormalLogs>;
template class EOSMultiTable<NQTLogs>;

} // namespace primitive