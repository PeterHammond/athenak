#ifndef EOS_PRIMITIVE_SOLVER_EOS_MULTITABLE_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_MULTITABLE_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_multitable.hpp
//  \brief Defines EOSMultiTable, which stores information from a tabulated
//         equation of state in MultiTable format, based on CompOSE.

///  \warning This code assumes the table to be uniformly spaced in
///           log ni, log t, and yj. 

/// For the avoidance of doubt, we use 'nb' to refer to the total 
/// conserved baryon number density, and 'n' for the number density in 
/// each subtable. 3D tables are stored first, then 2D.

#include <string>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"
#include "ps_types.hpp"
#include "eos_policy_interface.hpp"
#include "numtools_root.hpp"
#include "unit_system.hpp"
#include "logs.hpp"

namespace Primitive {

template<typename LogPolicy>
class EOSMultiTable : public EOSPolicyInterface, public LogPolicy, public SupportsEntropy {
  private:
    using LogPolicy::log2_;
    using LogPolicy::exp2_;
 public:
    enum PartialTableVariables {
      ECLOGP  = 0,  //! log (pressure / 1 MeV fm^-3)
      ECLOGE  = 1,  //! log (total energy density / 1 MeV fm^-3)
      ECENTD  = 2,  //! entropy density [kb fm^-3]
      ECDPDN  = 3,  //! derivative dp/dn [MeV]
      ECDPDT  = 4,  //! derivative dp/dT [fm^3]
      ECDSDN  = 5,  //! derivative ds/dn [kb]
      ECDSDT  = 6,  //! derivative ds/dT [kb MeV^-1 fm^-3]
      ECNVARS = 7
    };

    /// Read the table files.
    void ReadTableFromFile(std::string fname);
    
    /// Check if the EOS has been initialized properly.
    KOKKOS_INLINE_FUNCTION bool IsInitialized() const {
      return initialised;
    }

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    KOKKOS_INLINE_FUNCTION void SetNSpecies(int n) {
      // Number of species must be within limits
      assert (n<=MAX_SPECIES && n>=0);
    
      n_species = n;
      return;
    }

    /// Set the EOS unit system.
    KOKKOS_INLINE_FUNCTION void SetEOSUnitSystem(UnitSystem units) {
      eos_units = units;
    }

  protected:
    /// Constructor
    EOSMultiTable(): 
        nni("nni",1), nyi("nyi",1), nt("nt",1),
        inv_log_ni("inv_dlog_ni",1), inv_yi("inv_dyi",1), inv_log_t("inv_dlog_t",1),
        Pmin_fac("Pmin_fac",1), Pmin("Pmin",1),
        ni("ni",1), log_ni("log_ni",1),
        yi("yi",1),
        t("t",1), log_t("log_t",1),
        table("table", 1),
        offset_ni("offset_ni", 1), offset_yi("offset_yi", 1), offset_t("offset_t", 1), offset_table("offset_table", 1),
        y_weights("y_weights", 1, 1), n_weights("n_weights", 1, 1),
        t_union("T",1), log_t_union("T",1) {

      initialised = false;
      use_photons = false;

      n_tables_2D  = 0;
      n_tables_3D  = 0;
      n_ni_full    = 0;
      n_yi_full    = 0;
      n_T_full     = 0;
      n_table_full = 0;
      n_species    = 0;
      nt_union     = 0;

      eos_units = MakeNuclear();   

      min_h = std::numeric_limits<Real>::max();
      mb    = std::numeric_limits<Real>::quiet_NaN();
      min_n = std::numeric_limits<Real>::quiet_NaN();
      max_n = std::numeric_limits<Real>::quiet_NaN();
      min_T = std::numeric_limits<Real>::quiet_NaN();
      max_T = std::numeric_limits<Real>::quiet_NaN();
      for (int i = 0; i < MAX_SPECIES; i++) {
        min_Y[i] = std::numeric_limits<Real>::quiet_NaN();
        max_Y[i] = std::numeric_limits<Real>::quiet_NaN();
      }
    }

    /// Destructor
    ~EOSMultiTable() {}

    /// Calculate the energy density using.
    KOKKOS_INLINE_FUNCTION Real Energy(Real nb, Real T, const Real *Y) const {
      assert(initialised);
      Real result = 0.0;

      // 3D tables
      for (int i=0; i<n_tables_3D; ++i) {
        result += PartialEnergyDensity3D(i,nb,T,Y);
      }
        
      // 2D tables
        for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        result += PartialEnergyDensity2D(i,nb,T,Y);
      }

      // Photons
      if (use_photons) {
        result += photonEnergyConstant * pow(T,4);
      }

      return result;
    }

    /// Calculate the pressure using.
    KOKKOS_INLINE_FUNCTION Real Pressure(Real nb, Real T, Real *Y) const {
      assert(initialised);
      Real result = 0.0;

      // 3D Tables
      for (int i=0; i<n_tables_3D; ++i) {
        result += PartialPressure3D(i,nb,T,Y);
      }

      // 2D Tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        result += PartialPressure2D(i,nb,T,Y);
      }

      // Photons
      if (use_photons) {
        result += photonPressureConstant * pow(T,4);
      }

      return result;
    }

    /// Calculate the entropy per baryon using.
    KOKKOS_INLINE_FUNCTION Real Entropy(Real nb, Real T, Real *Y) const {
      assert(initialised);
      Real result = 0.0;

      // 3D Tables
      for (int i=0; i<n_tables_3D; ++i) {
        result += PartialEntropyDensity3D(i,nb,T,Y);
      }

      // 2D Tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        result += PartialEntropyDensity2D(i,nb,T,Y);
      }

      // Photons
      if (use_photons) {
        result += photonEntropyConstant * pow(T,3);
      }

      // Partials are entropy density (volumetric), so entropy per baryon requires division by nb
      return result / nb; 
    }

    /// Calculate the enthalpy per baryon using.
    KOKKOS_INLINE_FUNCTION Real Enthalpy(Real nb, Real T, Real *Y) const {
      Real const P = Pressure(nb, T, Y);
      Real const e = Energy(nb, T, Y);
      return (P + e)/nb;
    }

    /// Calculate the sound speed.
    KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real nb, Real T, Real *Y) const {
      assert(initialised);
      Real h = Enthalpy(nb, T, Y);
      Real dpdn = 0.0; // (ni/nb)*dPi/dni = dPi/dnb
      Real dpdT = 0.0; // dPi/dT
      Real dsdn = 0.0; // ni*dsi/dni - si = nb^2 * dSi/dnb
      Real dsdT = 0.0; // dsi/dT = nb * dSi/dT

      // 3D Tables
      for (int i=0; i<n_tables_3D; ++i) {
        dpdn += PartialDPDN3D(i,nb,T,Y);
        dpdT += PartialDPDT3D(i,nb,T,Y);
        dsdn += PartialDSDN3D(i,nb,T,Y);
        dsdT += PartialDSDT3D(i,nb,T,Y);
      }

      // 2D Tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        dpdn += PartialDPDN2D(i,nb,T,Y);
        dpdT += PartialDPDT2D(i,nb,T,Y);
        dsdn += PartialDSDN2D(i,nb,T,Y);
        dsdT += PartialDSDT2D(i,nb,T,Y);
      }

      // Photons
      if (use_photons) {
        dpdT += 4.0 * photonPressureConstant * pow(T,3.0);
        dsdn -= photonEntropyConstant * pow(T,3.0);
        dsdT += 3.0 * photonEntropyConstant * pow(T,2.0);
      }

      return (dpdn - dpdT*dsdn/(nb*dsdT))/h;
    }

    /// Calculate the specific internal energy per unit mass
    KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real nb, Real T, Real *Y) const {
      return Energy(nb, T, Y)/(mb*nb) - 1;
    }

    /* Chemical potentials are not yet implemented
    /// Calculate the baryon chemical potential
    KOKKOS_INLINE_FUNCTION Real BaryonChemicalPotential(Real nb, Real T, Real *Y);

    /// Calculate the charge chemical potential
    KOKKOS_INLINE_FUNCTION Real ChargeChemicalPotential(Real nb, Real T, Real *Y);

    /// Calculate the electron-lepton chemical potential
    KOKKOS_INLINE_FUNCTION Real ElectronLeptonChemicalPotential(Real nb, Real T, Real *Y);
    */

    /// Get the minimum enthalpy per baryon.
    KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const {
      return min_h;
    }

    /// Get the minimum pressure at a given density and composition
    KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real nb, Real *Y) const {
      return Pressure(nb, min_T, Y);
    }

    /// Get the maximum pressure at a given density and composition
    KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real nb, Real *Y) const {
      return Pressure(nb, max_T, Y);
    }

    /// Get the minimum energy at a given density and composition
    KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real nb, Real *Y) const {
      return Energy(nb, min_T, Y);
    }

    /// Get the maximum energy at a given density and composition
    KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real nb, Real *Y) const {
      return Energy(nb, max_T, Y);
    }

    /// Temperature from energy density
    KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real nb, Real e, Real *Y) const {
      assert (initialised);
      Real log_e = log2_(e);
      return TemperatureFromVar(ECLOGE, log_e, nb, Y);
    }

    /// Calculate the temperature using.
    KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real nb, Real p, Real *Y) const {
      assert (initialised);
      Real p_min = MinimumPressure(nb, Y);
      if (p <= p_min) {
        p = p_min;
        return min_T;
      } else {
        Real p_target = p;
        
        for (int i=0; i<n_tables_3D+n_tables_2D; ++i) {
          p_target += Pmin(i);
        }

        Real log_p = log2_(p_target);
        return TemperatureFromVar(ECLOGP, log_p, nb, Y);
      }
    }

  protected:
    /// Low level functions not intended for outside use
    // Read subtable files
    bool Read2DTableFromFile(std::string table_name);
    bool Read3DTableFromFile(std::string table_name);

    // Evaluation of subtables
    KOKKOS_INLINE_FUNCTION Real PartialPressure3D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return exp2_(eval_at_nty(table_idx, ECLOGP, ni, T, yi)) - Pmin(table_idx);
    }

    KOKKOS_INLINE_FUNCTION Real PartialPressure2D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return exp2_(eval_at_nt(table_idx, ECLOGP, ni, T)) - Pmin(table_idx);
    }

    KOKKOS_INLINE_FUNCTION Real PartialEnergyDensity3D(int table_idx, Real nb, Real T, const Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return exp2_(eval_at_nty(table_idx, ECLOGE, ni, T, yi));
    }

    KOKKOS_INLINE_FUNCTION Real PartialEnergyDensity2D(int table_idx, Real nb, Real T, const Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return exp2_(eval_at_nt(table_idx, ECLOGE, ni, T));
    }

    KOKKOS_INLINE_FUNCTION Real PartialEntropyDensity3D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return eval_at_nty(table_idx, ECENTD, ni, T, yi);
    }

    KOKKOS_INLINE_FUNCTION Real PartialEntropyDensity2D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return eval_at_nt(table_idx, ECENTD, ni, T);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDPDN3D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return (ni/nb)*eval_at_nty(table_idx, ECDPDN, ni, T, yi);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDPDN2D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return (ni/nb)*eval_at_nt(table_idx, ECDPDN, ni, T);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDPDT3D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return eval_at_nty(table_idx, ECDPDT, ni, T, yi);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDPDT2D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return eval_at_nt(table_idx, ECDPDT, ni, T);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDSDN3D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      Real s    = eval_at_nty(table_idx, ECENTD, ni, T, yi);
      Real dsdn = eval_at_nty(table_idx, ECDSDN, ni, T, yi);
      return ni*dsdn - s;
    }

    KOKKOS_INLINE_FUNCTION Real PartialDSDN2D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      Real s    = eval_at_nt(table_idx, ECENTD, ni, T);
      Real dsdn = eval_at_nt(table_idx, ECDSDN, ni, T);
      return ni*dsdn - s;
    }

    KOKKOS_INLINE_FUNCTION Real PartialDSDT3D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return eval_at_nty(table_idx, ECDSDT, ni, T, yi);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDSDT2D(int table_idx, Real nb, Real T, Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return eval_at_nt(table_idx, ECDSDT, ni, T);
    }

    // Temperature inversion
    KOKKOS_INLINE_FUNCTION Real TemperatureFromVar(int iv, Real var, Real nb, Real *Y) const {
      assert(initialised);
      assert(iv==ECLOGP || iv==ECLOGE || iv==ECENTD);

      // Indicies and weights
      int *in = new int[n_tables_3D + n_tables_2D]; 
      int *iy = new int[n_tables_3D];
      Real *wn0 = new Real[n_tables_3D + n_tables_2D];
      Real *wn1 = new Real[n_tables_3D + n_tables_2D];
      Real *wy0 = new Real[n_tables_3D];
      Real *wy1 = new Real[n_tables_3D];

      for (int i=0; i<n_tables_3D; ++i) {
        Real ni, yi;
        GetPartialInputs3D(i, nb, Y, ni, yi);

        weight_idx_ln(i, &(wn0[i]), &(wn1[i]), &(in[i]), log2_(ni));
        weight_idx_yi(i, &(wy0[i]), &(wy1[i]), &(iy[i]), yi);
      }

      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real ni;
        GetPartialInputs2D(i, nb, Y, ni);

        weight_idx_ln(i, &(wn0[i]), &(wn1[i]), &(in[i]), log2_(ni));
      }

      return 1.0;
      
      // TODO Fix
      /*
      auto f = [=](Real lt) {
        return RootFunction(lt+log_t_offset, var, iv, in, iy, wn0, wn1, wy0, wy1, log_t_offset);
      };

      int ilo = 0;
      int ihi = nt_union-1;
      Real lt_lo = log_t_union(ilo);
      Real lt_hi = log_t_union(ihi);

      Real flo = f(lt_lo);
      Real fhi = f(lt_hi);
      while (flo*fhi>0){
        if (ilo == ihi - 1) {
          break;
        } else {
          ilo += 1;
          flo = f(log_t_union(ilo));
        }
      }

      assert(flo*fhi <= 0);

      while (ihi - ilo > 1) {
        int ip = ilo + (ihi - ilo)/2;
        Real fp = f(log_t_union(ip));
        if (fp*flo <= 0) {
          ihi = ip;
          fhi = fp;
        }
        else {
          ilo = ip;
          flo = fp;
        }
      }

      assert(ihi - ilo == 1);

      Real lt_fp;
      Real lb = log_t_union(ilo)+log_t_offset;
      Real ub = log_t_union(ihi)+log_t_offset;

      bool result = root.FalsePosition(RootFunction, lb, ub, lt_fp, 1e-15, var, iv, in, iy, wn0, wn1, wy0, wy1, log_t_offset);
      lt_fp -= log_t_offset;

      assert(result);

      return exp2_(lt_fp);
      */
    }

    // Accessing tables
    KOKKOS_INLINE_FUNCTION void GetPartialInputs3D(int table_idx, Real nb, const Real *Y, Real &ni, Real &yi) const {
      Real Ni, Yi;
      GetPartialNi(table_idx, Y,Ni);
      GetPartialYi(table_idx, Y,Yi);
      yi = Yi/Ni;
      ni = Ni*nb;
      return;
    }

    KOKKOS_INLINE_FUNCTION void GetPartialInputs2D(int table_idx, Real nb, const Real *Y, Real &ni) const {
      Real Ni;
      GetPartialNi(table_idx, Y,Ni);
      ni = Ni*nb;
      return;
    }

    KOKKOS_INLINE_FUNCTION void GetPartialNi(int table_idx, const Real *Y, Real &Ni) const {
      Ni = n_weights(table_idx,0);
      for (int i=0; i<n_species; ++i) {
        Ni += n_weights(table_idx,1+i)*Y[i];
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION void GetPartialYi(int table_idx, const Real *Y, Real &Yi) const {
      Yi = y_weights(table_idx,0);
      for (int i=0; i<n_species; ++i) {
        Yi += y_weights(table_idx,1+i)*Y[i];
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION Real eval_at_nty(int table_idx, int vi, Real ni, Real T, Real Yi) const {
      return eval_at_lnty(table_idx, vi, log2_(ni), log2_(T), Yi);
    }

    KOKKOS_INLINE_FUNCTION Real eval_at_nt(int table_idx, int vi, Real ni, Real T) const {
      return eval_at_lnt(table_idx, vi, log2_(ni), log2_(T));
    }

    KOKKOS_INLINE_FUNCTION Real eval_at_lnty(int table_idx, int iv, Real ln, Real lt, Real yi) const {
      int in, iy, it;
      Real wn0, wn1, wy0, wy1, wt0, wt1;

      weight_idx_ln(table_idx, &wn0, &wn1, &in, ln);
      weight_idx_yi(table_idx, &wy0, &wy1, &iy, yi);
      weight_idx_lt(table_idx, &wt0, &wt1, &it, lt);

      return
        wn0 * (wy0 * (wt0 * table(index3D(table_idx, iv, in+0, iy+0, it+0))   +
                      wt1 * table(index3D(table_idx, iv, in+0, iy+0, it+1)))  +
               wy1 * (wt0 * table(index3D(table_idx, iv, in+0, iy+1, it+0))   +
                      wt1 * table(index3D(table_idx, iv, in+0, iy+1, it+1)))) +
        wn1 * (wy0 * (wt0 * table(index3D(table_idx, iv, in+1, iy+0, it+0))   +
                      wt1 * table(index3D(table_idx, iv, in+1, iy+0, it+1)))  +
               wy1 * (wt0 * table(index3D(table_idx, iv, in+1, iy+1, it+0))   +
                      wt1 * table(index3D(table_idx, iv, in+1, iy+1, it+1))));
    }

    KOKKOS_INLINE_FUNCTION Real eval_at_lnt(int table_idx, int iv, Real ln, Real lt) const {
      int in, it;
      Real wn0, wn1, wt0, wt1;

      weight_idx_ln(table_idx, &wn0, &wn1, &in, ln);
      weight_idx_lt(table_idx, &wt0, &wt1, &it, lt);

      return
        wn0 * (wt0 * table(index2D(table_idx, iv, in+0, it+0))   +
               wt1 * table(index2D(table_idx, iv, in+0, it+1)))  +
        wn1 * (wt0 * table(index2D(table_idx, iv, in+1, it+0))   +
               wt1 * table(index2D(table_idx, iv, in+1, it+1)));
    }

    KOKKOS_INLINE_FUNCTION void weight_idx_ln(int table_idx, Real *w0, Real *w1, int *in, Real ln) const {
      int offset = offset_ni(table_idx);
      if (ln<=log_ni(offset)) {
        *in = 0;
        *w0 = 1.0;
        *w1 = 0.0;
      } else if (ln>=log_ni(offset+nni(table_idx)-1)) {
        *in = nni(table_idx)-2;
        *w0 = 0.0;
        *w1 = 1.0;
      } else {
        *in = (ln - log_ni(offset))*inv_log_ni(table_idx);
        *w1 = (ln - log_ni(offset + (*in)))*inv_log_ni(table_idx);
        *w0 = 1.0 - (*w1);
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION void weight_idx_yi(int table_idx, Real *w0, Real *w1, int *iy, Real yi) const {
      int offset = offset_yi(table_idx);
      if (yi<=this->yi(offset)) {
        *iy = 0;
        *w0 = 1.0;
        *w1 = 0.0;
      } else if (yi>=this->yi(offset+nyi(table_idx)-1)) {
        *iy = nyi(table_idx)-2;
        *w0 = 0.0;
        *w1 = 1.0;
      } else {
        *iy = (yi - this->yi(offset))*inv_yi(table_idx);
        *w1 = (yi - this->yi(offset + (*iy)))*inv_yi(table_idx);
        *w0 = 1.0 - (*w1);
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION void weight_idx_lt(int table_idx, Real *w0, Real *w1, int *it, Real lt) const {
      int offset = offset_t(table_idx);
      if (lt<=log_t(offset)) {
        *it = 0;
        *w0 = 1.0;
        *w1 = 0.0;
      } else if (lt>=log_t(offset + nt(table_idx) - 1)) {
        *it = nt(table_idx)-2;
        *w0 = 0.0;
        *w1 = 1.0;
      } else{
        *it = (lt - log_t(offset))*inv_log_t(table_idx);
        *w1 = (lt - log_t(offset + (*it)))*inv_log_t(table_idx);
        *w0 = 1.0 - (*w1);
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION int index3D(int table_idx, int iv, int in, int iy, int it) const {
      // return offset_table(table_idx) + iv*(nni(table_idx)*nyi(table_idx)*nt(table_idx)) + in*(nyi(table_idx)*nt(table_idx)) + iy*(nt(table_idx)) + it;
      return offset_table(table_idx) + it + nt(table_idx)*(iy + nyi(table_idx)*(in + nni(table_idx)*iv));
    }

    KOKKOS_INLINE_FUNCTION int index2D(int table_idx, int iv, int in, int it) const {
      // return offset_table(table_idx) + iv*(nni(table_idx)*nt(table_idx)) + in*(nt(table_idx)) + it;
      return offset_table(table_idx) + it + nt(table_idx)*(in + nni(table_idx)*iv);
    }

    // Minimum enthalpy per baryon
    Real min_h;

    // Protect against using before loading
    bool initialised;

    // Photons
    bool use_photons;
    
    // Constants for photons
    Real pi   = 3.1415926535897932;
    Real h_SI = 6.62607015e-34;  // Exact in J s
    Real c_SI = 299792458.0;     // Exact in m s^-1
    Real e_SI = 1.602176634e-19; // Exact in C

    Real hc_SI = h_SI*c_SI; // J m
    Real MeV_SI = 1.0e6*e_SI; // J per MeV
    Real hc_MeV = hc_SI * 1.0e15 / MeV_SI;

    Real photonEnergyConstant = 8.0 * pow(pi,5) / (15.0 * pow(hc_MeV,3.0));
    Real photonPressureConstant = photonEnergyConstant/3.0;
    Real photonEntropyConstant = 4.0*photonPressureConstant;

    // Subtables
    int n_tables_3D;
    int n_tables_2D;
    int n_ni_full;
    int n_yi_full;
    int n_T_full;
    int n_table_full;

    // Table storage
    DvceArray1D<int> nni, nyi, nt;                   // <number density, fraction, temperature> samples for each subtable
    DvceArray1D<Real> inv_log_ni, inv_yi, inv_log_t; // inverse <log number density, fraction, log temperature> spacing for each subtable
    DvceArray1D<Real> Pmin_fac, Pmin;                // pressure offsets for where Pi<=0

    // Sequential table storage. 
    DvceArray1D<Real> ni, log_ni;                                  // <number density, log number density> for each subtable sequentially
    DvceArray1D<Real> yi;                                          // fractions for each subtable sequentially
    DvceArray1D<Real> t, log_t;                                    // <temperature, log temperature> for each subtable sequentially
    DvceArray1D<Real> table;                                       // data for each subtable sequentially
    DvceArray1D<int> offset_ni, offset_yi, offset_t, offset_table; // offsets for start of each subtables data

    DvceArray2D<int> y_weights, n_weights; // weights for calculating ni and yi from nb and Y


    // Union of temperatures for rootsolver
    DvceArray1D<Real> t_union, log_t_union;
    int nt_union;

    // TODO Fix
    /*
    /// The root solver.
    class RootFunctor {
      public:
        KOKKOS_INLINE_FUNCTION Real operator()(Real lt_o, Real var, int iv, int* in, int* iy, Real *wn0, Real* wn1, Real* wy0, Real* wy1, Real log_t_offset) const {
          Real var_pt = 0.0;
          Real lt = lt_o - log_t_offset;

          // 3D Tables
          for (int i=0; i<n_tables_3D; ++i) {
            Real wt0, wt1;
            int it;
            weight_idx_lt(i,&wt0, &wt1, &it, lt);
            var_pt += exp2_(wn0[i] * (wy0[i] * (wt0 * table(index3D(i, iv, in[i]+0, iy[i]+0, it+0))   +
                                                wt1 * table(index3D(i, iv, in[i]+0, iy[i]+0, it+1)))  +
                                      wy1[i] * (wt0 * table(index3D(i, iv, in[i]+0, iy[i]+1, it+0))   +
                                                wt1 * table(index3D(i, iv, in[i]+0, iy[i]+1, it+1)))) +
                            wn1[i] * (wy0[i] * (wt0 * table(index3D(i, iv, in[i]+1, iy[i]+0, it+0))   +
                                                wt1 * table(index3D(i, iv, in[i]+1, iy[i]+0, it+1)))  +
                                      wy1[i] * (wt0 * table(index3D(i, iv, in[i]+1, iy[i]+1, it+0))   +
                                                wt1 * table(index3D(i, iv, in[i]+1, iy[i]+1, it+1)))));
          }

          // 2D Tables
          for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
            Real wt0, wt1;
            int it;
            weight_idx_lt(i,&wt0, &wt1, &it, lt);
            var_pt += exp2_(wn0[i] * (wt0 * table(index2D(i, iv, in[i]+0, it+0))   +
                                      wt1 * table(index2D(i, iv, in[i]+0, it+1)))  +
                            wn1[i] * (wt0 * table(index2D(i, iv, in[i]+1, it+0))   +
                                      wt1 * table(index2D(i, iv, in[i]+1, it+1))));
          }

          return var - log2_(var_pt);
        }
    };

    NumTools::Root root;
    RootFunctor RootFunction;
    */
    Real log_t_offset=10.0;

};

}; // namespace Primitive


#endif // EOS_PRIMITIVE_SOLVER_EOS_MULTITABLE_HPP_