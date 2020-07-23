//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file interface_physics.cpp
//  \brief 

#include <iostream>

#include "parameter_input.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif


using namespace hydro;
//----------------------------------------------------------------------------------------
// \fn Mesh::SelectPhysics()

void Mesh::SelectPhysics(std::unique_ptr<ParameterInput> &pin) {

  // parse input blocks to see which physics defined
  bool hydro_defined = pin->DoesBlockExist("hydro");

  // loop through MBs on this rank and construct physics modules and tasks lists
  for (auto &mb : mblocks) {

    // physics modules
    if (hydro_defined) {
      mb.phydro = new hydro::Hydro(&mb, pin);
    } else {
      mb.phydro = nullptr;
      std::cout << "Hydro block not found in input file" << std::endl;
    }

    // task lists
    if (mb.phydro != nullptr) mb.phydro->HydroAddTasks(mb.tl_onestage);

  } // end loop over MBs

  return;
}

//----------------------------------------------------------------------------------------
// \fn Mesh::NewTimeStep()

void Mesh::NewTimeStep(const Real tlim) {

  // limit increase in timestep to 2x old value
  dt = 2.0*dt;

  // cycle over all MeshBlocks on this rank and find minimum dt
  for (const auto &mb : mblocks) { dt = std::min(dt, (cfl_no)*(mb.phydro->dtnew) ); }

  // TODO: get minimum dt over all MPI ranks

  // limit last time step to stop at tlim *exactly*
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}

  return;
}
