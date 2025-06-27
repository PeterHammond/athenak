//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_compose.cpp
//  \brief Implementation of EOSMultiTable

#include "eos_multitable.hpp"

using namespace Primitive;

template<typename LogPolicy>
void EOSMultiTable<LogPolicy>::ReadTableFromFile(std::string fname) {
  if (initialised==false) {
    // TODO

    initialised = true;
  }
}

template<typename LogPolicy>
void EOSMultiTable<LogPolicy>::Read2DTableFromFile(std::string fname) {
  //TODO
}


template<typename LogPolicy>
void EOSMultiTable<LogPolicy>::Read3DTableFromFile(std::string fname) {
  //TODO
}

