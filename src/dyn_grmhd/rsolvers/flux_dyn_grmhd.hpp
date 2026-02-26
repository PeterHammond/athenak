#ifndef DYN_GRMHD_RSOLVERS_FLUX_DYN_GRMHD_HPP_
#define DYN_GRMHD_RSOLVERS_FLUX_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file flux_dyngrmhd.hpp
//! \brief Calculate left and right fluxes for a central scheme in GRMHD
#include <stdio.h>
#include <math.h>

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void SingleStateFlux
//! \brief inline function for calculating GRMHD fluxes

template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void SingleStateFlux(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    Real prim_l[NPRIM], Real prim_r[NPRIM], Real Bu_l[NPRIM], Real Bu_r[NPRIM],
    const int nmhd, const int nscal,
    Real g3d[NSPMETRIC], Real beta_u[3], Real alpha,
    Real cons_l[NCONS], Real cons_r[NCONS],
    Real flux_l[NCONS], Real flux_r[NCONS], Real bflux_l[NMAG], Real bflux_r[NMAG],
    Real& bsql, Real& bsqr) {
  constexpr int pvx = PVX + (ivx - IVX);
  constexpr int pvy = PVX + ((ivx - IVX) + 1)%3;
  constexpr int pvz = PVX + ((ivx - IVX) + 2)%3;

  constexpr int csx = CSX + (ivx - IVX);

  constexpr int ibx = ivx - IVX;
  constexpr int iby = ((ivx - IVX) + 1)%3;
  constexpr int ibz = ((ivx - IVX) + 2)%3;

  const Real ialpha = 1.0/alpha;

  // Calculate conserved variables
  eos.ps.PrimToCon(prim_l, cons_l, Bu_l, g3d);
  eos.ps.PrimToCon(prim_r, cons_r, Bu_r, g3d);

  // Calculate W for the left state. 
  // PH: These are only used per-side, can be re-used for rhs
  Real uu_s[3] = {prim_l[IVX], prim_l[IVY], prim_l[IVZ]};
  Real ud_s[3];
  Primitive::LowerVector(ud_s, uu_s, g3d);
  Real iWsq_s = 1.0/(1.0 + Primitive::Contract(uu_s, ud_s));
  Real iW_s = sqrt(iWsq_s);
  Real vc_s = prim_l[pvx]*iW_s - beta_u[ivx-IVX]*ialpha;

  // Calculate 4-magnetic field for the left state.
  // PH: Same as above
  Real bu0_s = Primitive::Contract(Bu_l, ud_s)*ialpha;
  Real bd_s[3], Bd_s[3];
  Primitive::LowerVector(Bd_s, Bu_l, g3d);
  for (int a = 0; a < 3; a++) {
    bd[a] = (alpha*bu0_s*ud_s[a] + Bd_s[a])*iW_s;
  }
  bsql = (Primitive::SquareVector(Bu_l, g3d) + SQR(alpha*bu0_s))*iWsq_s;

  // Calculate fluxes for the left state.
  flux_l[CDN] = cons_l[CDN]*vc_s;
  flux_l[CSX] = (cons_l[CSX]*vc_s - bd[0]*Bu_l[ibx]*iW_s);
  flux_l[CSY] = (cons_l[CSY]*vc_s - bd[1]*Bu_l[ibx]*iW_s);
  flux_l[CSZ] = (cons_l[CSZ]*vc_s - bd[2]*Bu_l[ibx]*iW_s);
  flux_l[csx] += (prim_l[PPR] + 0.5*bsql);
  flux_l[CTA] = (cons_l[CTA]*vc_s - alpha*bu0_s*Bu_l[ibx]*iW_s
          + (prim_l[PPR] + 0.5*bsql)*prim_l[ivx]*iW_s);

  bflux_l[ibx] = 0.0;
  bflux_l[iby] = (Bu_l[iby]*vc_s -
                    Bu_l[ibx]*(prim_l[pvy]*iW_s - beta_u[pvy - PVX]*ialpha));
  bflux_l[ibz] = (Bu_l[ibz]*vc_s -
                    Bu_l[ibx]*(prim_l[pvz]*iW_s - beta_u[pvz - PVX]*ialpha));

  // Calculate W for the right state.
  // PH: re-use temporaries from lhs
  uu_s[0] = prim_r[IVX]; uu_s[1] = prim_r[IVY]; uu_s[2] = prim_r[IVZ];
  Primitive::LowerVector(ud_s, uu_s, g3d);
  iWsq_s = 1.0/(1.0 + Primitive::Contract(uu_s, ud_s));
  iW_s = sqrt(iWsq_s);
  vc_s = prim_r[pvx]*iW_s - beta_u[ivx-IVX]*ialpha;

  // Calculate 4-magnetic field for the right state.
  bu0_s = Primitive::Contract(Bu_r, ud_s)*ialpha;
  Primitive::LowerVector(Bd_s, Bu_r, g3d);
  for (int a = 0; a < 3; a++) {
    bd_s[a] = (alpha*bu0_s*ud_s[a] + Bd_s[a])*iW_s;
  }
  bsqr = (Primitive::SquareVector(Bu_r, g3d) + SQR(alpha*bu0_s))*iWsq_s;

  // Calculate fluxes for the right state.
  flux_r[CDN] = cons_r[CDN]*vc_s;
  flux_r[CSX] = (cons_r[CSX]*vc_s - bdr[0]*Bu_r[ibx]*iW_s);
  flux_r[CSY] = (cons_r[CSY]*vc_s - bdr[1]*Bu_r[ibx]*iW_s);
  flux_r[CSZ] = (cons_r[CSZ]*vc_s - bdr[2]*Bu_r[ibx]*iW_s);
  flux_r[csx] += (prim_r[PPR] + 0.5*bsqr);
  flux_r[CTA] = (cons_r[CTA]*vc_s - alpha*bu0_s*Bu_r[ibx]*iW_s
          + (prim_r[PPR] + 0.5*bsqr)*prim_r[ivx]*iW_s);

  bflux_r[ibx] = 0.0;
  bflux_r[iby] = (Bu_r[iby]*vc_s -
                    Bu_r[ibx]*(prim_r[pvy]*iW_s - beta_u[pvy - PVX]*ialpha));
  bflux_r[ibz] = (Bu_r[ibz]*vc_s -
                    Bu_r[ibx]*(prim_r[pvz]*iW_s - beta_u[pvz - PVX]*ialpha));
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_FLUX_DYN_GRMHD_HPP_
