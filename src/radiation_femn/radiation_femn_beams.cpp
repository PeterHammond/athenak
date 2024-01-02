//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_beams.cpp
//  \brief set up beam sources

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "coordinates/cell_locations.hpp"
#include "radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

TaskStatus RadiationFEMN::BeamsSourcesFEMN(Driver *pdriver, int stage) {

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int npts1 = num_points_total - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;

  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  auto &f0_ = pmy_pack->pradfemn->f0;

  par_for("radiation_femn_beams_populate", DevExeSpace(), 0, nmb1, 0, npts1, 0, (n3 - 1), 0, (n2 - 1),
          KOKKOS_LAMBDA(int m, int n, int k, int j) {
            switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
              case BoundaryFlag::outflow:
                for (int i = 0; i < ng; ++i) {
                  f0_(m, n, k, j, is - i - 1) = 1.;
                }
                break;
              default:break;
            }
          });

  return TaskStatus::complete;
}

TaskStatus RadiationFEMN::BeamsSourcesFPN(Driver *pdriver, int stage) {

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int npts1 = num_points_total - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &size = pmy_pack->pmb->mb_size;

  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  auto &f0_ = pmy_pack->pradfemn->f0;

  par_for("radiation_femn_beams_populate_fpn", DevExeSpace(), 0, nmb1, 0, npts1, 0, (n3 - 1), 0, (n2 - 1),
          KOKKOS_LAMBDA(int m, int n, int k, int j) {

            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            int nx2 = indcs.nx2;
            Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

            switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
              case BoundaryFlag::outflow:

                if (beam_source_1_y1 <= x2 && x2 <= beam_source_1_y2) {
                  for (int i = 0; i < ng; ++i) {
                    f0_(m, n, k, j, is - i - 1) = beam_source_1_vals(n);
                  }
                }

                if (num_beams > 1 && beam_source_2_y1 <= x2 && x2 <= beam_source_2_y2) {
                  for (int i = 0; i < ng; ++i) {
                    f0_(m, n, k, j, is - i - 1) = beam_source_2_vals(n);
                  }
                }
                break;

              default:break;
            }
          });

  return TaskStatus::complete;
}

void RadiationFEMN::InitializeBeamsSourcesFPN() {

  std::cout << "Initializing beam sources for FPN" << std::endl;
  for (int i = 0; i < num_points; i++) {
    beam_source_1_vals(i) = FPNBasis(angular_grid(i, 0), angular_grid(i, 1), beam_source_1_phi, beam_source_1_theta);
  }

  if (num_beams > 1) {
    for (int i = 0; i < num_points; i++) {
      beam_source_2_vals(i) = FPNBasis(angular_grid(i, 0), angular_grid(i, 1), beam_source_2_phi, beam_source_2_theta);
    }
  }
}

void RadiationFEMN::InitializeBeamsSourcesFEMN() {

  std::cout << "Initializing beam sources for FEMN" << std::endl;

}

}