//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_compose_test.cpp
//  \brief Unit test for EOSCompOSE to make sure it works properly.

#include <sstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

template<class LogPolicy>
void PerformTests(Mesh* pmesh, ParameterInput *pin);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSCompOSE unit test only works for DynGRMHD!\n";
    exit(EXIT_FAILURE);
  }

  std::string eos_string = pin->GetString("mhd", "dyn_eos");

  if (eos_string.compare("multitable") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSMultiTable unit test needs mhd/dyn_eos = multitable!\n";
    exit(EXIT_FAILURE);
  }

  bool use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT", false);

  if (use_NQT) {
    PerformTests<Primitive::NQTLogs>(pmy_mesh_, pin);
  } else {
    PerformTests<Primitive::NormalLogs>(pmy_mesh_, pin);
  }

  std::cout << "Test Passed!\n";

  // This is needed to initialize the ADM variables to Minkowski. Otherwise the pgen
  // will have a bunch of C2P failures at the end.
  pmbp->padm->SetADMVariables(pmbp);

  return;
}

template<class LogPolicy>
void PerformTests(Mesh *pmesh, ParameterInput *pin) {
  MeshBlockPack *pmbp = pmesh->pmb_pack;

  // Commit a crime against humanity to get access to the EOS
  Primitive::EOS<Primitive::EOSMultiTable<LogPolicy>, Primitive::ResetFloor>& eos =
    static_cast<
      dyngr::DynGRMHDPS<
        Primitive::EOSMultiTable<LogPolicy>,
        Primitive::ResetFloor
      >*
    >(pmbp->pdyngr)->eos.ps.GetEOSMutable();

  // Get the range of the table
  LogPolicy logs;
  Real nmin = eos.GetMinimumDensity();
  Real nmax = eos.GetMaximumDensity();
  Real lnmin = logs.log2_(nmin);
  Real lnmax = logs.log2_(nmax);

  int nscalars = pin->GetOrAddInteger("mhd","nscalars",0);
  Real Ymin[MAX_SPECIES];
  Real Ymax[MAX_SPECIES];
  for (int r=0; r<nscalars; ++r) {
    Ymin[r] = eos.GetMinimumSpeciesFraction(r);
    Ymax[r] = eos.GetMaximumSpeciesFraction(r);
  }
  
  Real Tmin = eos.GetMinimumTemperature();
  Real Tmax = eos.GetMaximumTemperature();
  Real lTmin = logs.log2_(Tmin);
  Real lTmax = logs.log2_(Tmax);

  int nn = pin->GetOrAddInteger("problem", "nn", 100);
  int nY = pin->GetOrAddInteger("problem", "nY", 100);
  int nT = pin->GetOrAddInteger("problem", "nT", 100);

  Real dln = (lnmax - lnmin) / (nn - 1);
  Real dY[MAX_SPECIES];
  for (int r=0; r<nscalars; ++r) {
    dY[r] = (Ymax[r] - Ymin[r]) / (nY - 1);
  }
  Real dlT = (lTmax - lTmin) / (nT - 1);

  // To make sure things are working as intended, we want to test what happens when things
  // are below and above the ranges of the table.
  int inlo = -1;
  int inhi = nn;
  int iYlo[MAX_SPECIES];
  int iYhi[MAX_SPECIES];
  for (int r=0; r<nscalars; ++r) {
    iYlo[r] = -1;
    iYhi[r] = nY;
  }
  int iTlo = -1;
  int iThi = nT;

  bool global_success = true;

  Real tol = static_cast<Real>(std::numeric_limits<float>::epsilon());

  // const int ni = (iThi - iTlo + 1);
  // const int nji = (iYhi - iYlo + 1)*ni;
  // const int nkji = (inhi - inlo + 1)*nji;

  int Nkji[MAX_SPECIES+2];
  Nkji[0] = (iThi - iTlo + 1);
  for (int r=0; r<nscalars; ++r) {
    Nkji[r+1] = (iYhi[r] - iYlo[r] + 1)*Nkji[r];
  }
  Nkji[nscalars+1] = (inhi - inlo + 1)*Nkji[nscalars];
  const int iteration_count = Nkji[nscalars+1];

  // Check the table's ability to handle an exact conversion.
  Kokkos::parallel_reduce("pgen_test", Kokkos::RangePolicy<>(DevExeSpace(), 0, iteration_count),
  KOKKOS_LAMBDA(const int &idx, bool &success) {
    int iT = (idx%Nkji[0]) + iTlo;
    int iY[MAX_SPECIES];
    for (int r=0; r<nscalars; ++r) {
      iY[r] = ((idx%Nkji[r+1])/Nkji[r]) + iYlo[r];
    }
    int in = (idx/Nkji[nscalars]) + inlo;

    // Calculate the table input.
    // Note that we do *NOT* clamp the input values to the table ranges. The table
    // frequently gets slightly invalid units, and it needs to be able to deal with them
    // in a sensible way.
    Real ln = lnmin + in*dln;
    Real Y[MAX_SPECIES] = {0.0};
    for (int r=0; r<nscalars; ++r) {
      Y[r] = Ymin[r] + iY[r]*dY[r];
    }
    Real lT = lTmin + iT*dlT;

    // Sanitise scalars
    eos.ApplySpeciesLimits(Y);
    
    Real n = logs.exp2_(ln);
    Real T = logs.exp2_(lT);
    // Try to calculate the pressure and energy. We don't do anything with the pressure
    // (since it's not guaranteed to be monotonic), but this checks that it will get
    // calculated without failing.
    Real P = eos.GetPressure(n, T, Y);
    Real e = eos.GetEnergy(n, T, Y);

    // Try to invert the energy to get temperature
    Real T_test = eos.GetTemperatureFromE(n, e, Y);
    Real e_test = eos.GetEnergy(n, T_test, Y);

    // Check the error on T
    Real error = T_test/T - 1.;
    if (Kokkos::fabs(error) > tol) {
      // Check if the failure was because we were outside the table.
      bool outside_table = n < nmin || n > nmax || T < Tmin || T > Tmax;
      for (int r=0; r<nscalars; ++r) {
        outside_table = outside_table || Y[r] < Ymin[r] || Y[r] > Ymax[r];
      }
      if (!outside_table) {
        Kokkos::printf("The following point was recovered poorly:\n"
                       "  n = %20.17g\n"
                       "  Y = %20.17g\n"
                       "  Y = %20.17g\n"
                       "  T = %20.17g\n"
                       "  e = %20.17g\n"
                       "  p = %20.17g\n"
                       "Calculated temperature:\n"
                       "  T_test = %20.17g\n"
                       "  e_test = %20.17g\n"
                       "  T_error = %20.17g\n",
                       n, Y[0], Y[1], T, e, P, T_test, e_test, error);
        success = false;
      } else if ( (logs.log2_(T_test) < lTmin) || (logs.log2_(T_test) > lTmax)) {
        Kokkos::printf("The following point recovers an invalid temperature:\n"
                       "  n = %20.17g\n"
                       "  Y = %20.17g\n"
                       "  Y = %20.17g\n"
                       "  T = %20.17g\n"
                       "  e = %20.17g\n"
                       "  p = %20.17g\n"
                       "Calculated temperature:\n"
                       "  T_test = %20.17g\n"
                       "  Tmin = %20.17g\n"
                       "  Tmax = %20.17g\n",
                       n, Y[0], Y[1], T, e, P, T_test, logs.exp2_(lTmin), logs.exp2_(lTmax));
        success = false;
      }
    }
  }, Kokkos::LAnd<bool>(global_success));

  // Check the table's ability to recover the temperature correctly when the energy or
  // pressure falls below the zero-temperature limit. We adjust the bounds of density and
  // Y to be physical; they should already be physical by this point.
  bool pert_success = true;

  inlo = 0;
  inhi = nn - 1;
  for (int r=0; r<nscalars; ++r) {
    iYlo[r] = 0;
    iYhi[r] = nY - 1;
  }

  Nkji[0] = (inhi - inlo + 1);
  for (int r=0; r<nscalars; ++r) {
    Nkji[r+1] = (iYhi[r] - iYlo[r] + 1)*Nkji[r];
  }
  const int iteration_count_0T = Nkji[nscalars];


  Kokkos::parallel_reduce("pgen_test", Kokkos::RangePolicy<>(DevExeSpace(), 0, iteration_count_0T),
  KOKKOS_LAMBDA(const int &idx, bool &success) {
    int in = (idx%Nkji[0]) + inlo;
    int iY[MAX_SPECIES];
    for (int r=0; r<nscalars; ++r) {
      iY[r] = ((idx%Nkji[r+1])/Nkji[r]) + iYlo[r];
    }

    // Calculate the table input assuming zero temperature.
    Real ln = lnmin + in*dln;
    Real Y[MAX_SPECIES] = {0.0};
    for (int r=0; r<nscalars; ++r) {
      Y[r] = Ymin[r] + iY[r]*dY[r];
    }
    Real lT = lTmin;

    Real n = logs.exp2_(ln);
    Real T = logs.exp2_(lT);

    // Try to calculate the pressure and energy.
    Real P = eos.GetPressure(n, T, Y);
    Real e = eos.GetEnergy(n, T, Y);

    // Perturb both the pressure and the energy downward a significant amount.
    Real P_pert = 0.5*P;
    Real e_pert = 0.5*e;

    // Check that we recover the minimum temperature.
    Real T_p = eos.GetTemperatureFromP(n, P_pert, Y);
    Real T_e = eos.GetTemperatureFromE(n, e_pert, Y);

    Real error_p = T_p/T - 1.;
    Real error_e = T_e/T - 1.;
    if (Kokkos::fabs(error_p) > tol) {
      Kokkos::printf("The temperature was not recovered correctly from pressure:\n" // NOLINT
                     "  n = %20.17g\n"
                     "  Y = %20.17g\n"
                     "  T = %20.17g\n"
                     "Calculated temperature:\n"
                     "  T_test = %20.17g\n"
                     "  error = %20.17g\n",
                     n, Y[0], T, T_p, error_p);
      success = false;
    }
    if (Kokkos::fabs(error_e) > tol) {
      Kokkos::printf("The temperature was not recovered correctly from energy:\n" // NOLINT
                     "  n = %20.17g\n"
                     "  Y = %20.17g\n"
                     "  T = %20.17g\n"
                     "Calculated temperature:\n"
                     "  T_test = %20.17g\n"
                     "  error = %20.17g\n",
                     n, Y[0], T, T_e, error_e);
      success = false;
    }
  }, Kokkos::LAnd<bool>(pert_success));

  global_success = global_success && pert_success;

  if (!global_success) {
    std::cout << "The test was not successful...\n";
    exit(EXIT_FAILURE);
  }

  return;
}
