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
/// conserved baryon number density, and 'ni' for the number density in 
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
    void ReadTableFromFile(std::string dname, std::string fname);
    
    /// Check if the EOS has been initialized properly.
    KOKKOS_INLINE_FUNCTION bool IsInitialized() const {
      return initialised;
    }

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    KOKKOS_INLINE_FUNCTION void SetNSpecies(const int n) {
      // Number of species must be within limits
      assert (n<=MAX_SPECIES && n>=0);
    
      n_species = n;
      return;
    }

    KOKKOS_INLINE_FUNCTION void SetUsePhotons(const bool photons) {
      use_photons = photons;
      return;
    }

    /// Set the EOS unit system.
    KOKKOS_INLINE_FUNCTION void SetEOSUnitSystem(const UnitSystem units) {
      eos_units = units;
      return;
    }

  protected:
    /// Constructor
    EOSMultiTable(): 
        nni("nni",1), nyi("nyi",1),
        inv_dlog_ni("inv_dlog_ni",1), inv_dyi("inv_dyi",1),
        Pmin("Pmin",1),
        ni("ni",1), log_ni("log_ni",1),
        yi("yi",1),
        table("table", 1),
        offset_ni("offset_ni", 1), offset_yi("offset_yi", 1), offset_table("offset_table", 1),
        y_weights("y_weights", 1, 1), n_weights("n_weights", 1, 1),
        t_shared("T",1), log_t_shared("log_T",1) {

      initialised = false;
      use_photons = false;

      n_tables_2D  = 0;
      n_tables_3D  = 0;
      n_ni_full    = 0;
      n_yi_full    = 0;
      n_table_full = 0;
      n_species    = 0;
      n_t_shared   = 0;

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
      dlog_t_shared = std::numeric_limits<Real>::quiet_NaN();
      inv_dlog_t_shared = std::numeric_limits<Real>::quiet_NaN();

      Pmin_fac = 1.0e-10;
      // We don't want to root solver to fail unless something goes horribly wrong.
      // Worst case scenario is bisection every other step, so for tol=1e-15
      // the maximum number of steps should be:
      // log_2(10)*15*2 \approx 100
      root.iterations = 100;
    }


    /// Destructor
    ~EOSMultiTable() {}

    /// Calculate the energy density using.
    KOKKOS_INLINE_FUNCTION Real Energy(const Real nb, const Real T, const Real *Y) const {
      assert(initialised);
      Real result = 0.0;
      Real lt = log2_(T);

      int it;
      Real wt1;
      weight_idx_lt(&wt1, &it, lt);

      // 3D tables
      for (int i=0; i<n_tables_3D; ++i) {
        Real ni, yi;
        GetPartialInputs3D(i, nb, Y, ni, yi);
        Real lni = log2_(ni);
        
        int in, iy;
        Real wn1, wy1;

        weight_idx_ln(i, &wn1, &in, lni);
        weight_idx_yi(i, &wy1, &iy, yi);

        result += exp2_(eval_at_inty(i, ECLOGE, in, it, iy, wn1, wt1, wy1));
      }
        
      // 2D tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real ni;
        GetPartialInputs2D(i, nb, Y, ni);
        Real lni = log2_(ni);
        
        int in;
        Real wn1;

        weight_idx_ln(i, &wn1, &in, lni);

        result += exp2_(eval_at_int(i, ECLOGE, in, it, wn1, wt1));
      }

      // Photons
      if (use_photons) {
        result += photonEnergyConstant * pow(T,4);
      }

      return result;
    }

    /// Calculate the pressure using.
    KOKKOS_INLINE_FUNCTION Real Pressure(const Real nb, const Real T, const Real *Y) const {
      assert(initialised);
      Real result = 0.0;
      Real lt = log2_(T);

      int it;
      Real wt1;
      weight_idx_lt(&wt1, &it, lt);

      // 3D Tables
      for (int i=0; i<n_tables_3D; ++i) {
        Real ni, yi;
        GetPartialInputs3D(i, nb, Y, ni, yi);
        Real lni = log2_(ni);
        
        int in, iy;
        Real wn1, wy1;

        weight_idx_ln(i, &wn1, &in, lni);
        weight_idx_yi(i, &wy1, &iy, yi);

        result += exp2_(eval_at_inty(i, ECLOGP, in, it, iy, wn1, wt1, wy1)) - Pmin(i);
      }

      // 2D Tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real ni;
        GetPartialInputs2D(i, nb, Y, ni);
        Real lni = log2_(ni);
        
        int in;
        Real wn1;

        weight_idx_ln(i, &wn1, &in, lni);

        result += exp2_(eval_at_int(i, ECLOGP, in, it, wn1, wt1)) - Pmin(i);
      }

      // Photons
      if (use_photons) {
        result += photonPressureConstant * pow(T,4);
      }

      return result;
    }

    /// Calculate the entropy per baryon using.
    KOKKOS_INLINE_FUNCTION Real Entropy(const Real nb, const Real T, const Real *Y) const {
      assert(initialised);
      Real result = 0.0;
      Real lt = log2_(T);

      int it;
      Real wt1;
      weight_idx_lt(&wt1, &it, lt);

      // 3D Tables
      for (int i=0; i<n_tables_3D; ++i) {
        Real ni, yi;
        GetPartialInputs3D(i, nb, Y, ni, yi);
        Real lni = log2_(ni);
        
        int in, iy;
        Real wn1, wy1;

        weight_idx_ln(i, &wn1, &in, lni);
        weight_idx_yi(i, &wy1, &iy, yi);

        result += eval_at_inty(i, ECENTD, in, it, iy, wn1, wt1, wy1);
      }

      // 2D Tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real ni;
        GetPartialInputs2D(i, nb, Y, ni);
        Real lni = log2_(ni);
        
        int in;
        Real wn1;

        weight_idx_ln(i, &wn1, &in, lni);

        result += eval_at_int(i, ECENTD, in, it, wn1, wt1);
      }

      // Photons
      if (use_photons) {
        result += photonEntropyConstant * pow(T,3);
      }

      // Partials are entropy density (volumetric), so entropy per baryon requires division by nb
      return result / nb; 
    }

    /// Calculate the enthalpy per baryon using.
    KOKKOS_INLINE_FUNCTION Real Enthalpy(const Real nb, const Real T, const Real *Y) const {
      assert(initialised);
      Real result = 0.0;
      Real lt = log2_(T);

      int it;
      Real wt1;
      weight_idx_lt(&wt1, &it, lt);

      // 3D Tables
      for (int i=0; i<n_tables_3D; ++i) {
        Real ni, yi;
        GetPartialInputs3D(i, nb, Y, ni, yi);
        Real lni = log2_(ni);
        
        int in, iy;
        Real wn1, wy1;

        weight_idx_ln(i, &wn1, &in, lni);
        weight_idx_yi(i, &wy1, &iy, yi);

        result += exp2_(eval_at_inty(i, ECLOGP, in, it, iy, wn1, wt1, wy1)) - Pmin(i);
        result += exp2_(eval_at_inty(i, ECLOGE, in, it, iy, wn1, wt1, wy1));
      }

      // 2D Tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real ni;
        GetPartialInputs2D(i, nb, Y, ni);
        Real lni = log2_(ni);
        
        int in;
        Real wn1;

        weight_idx_ln(i, &wn1, &in, lni);

        result += exp2_(eval_at_int(i, ECLOGP, in, it, wn1, wt1)) - Pmin(i);
        result += exp2_(eval_at_int(i, ECLOGE, in, it, wn1, wt1));
      }

      // Photons
      if (use_photons) {
        result += (photonPressureConstant + photonEnergyConstant) * pow(T,4);
      }

      return result/nb;
    }

    /// Calculate the sound speed.
    KOKKOS_INLINE_FUNCTION Real SoundSpeed(const Real nb, const Real T, const Real *Y) const {
      assert(initialised);
      // N.B. all the extra nb factors cancel in the final eqn for cs2
      Real h = 0.0;     // h = p+e N.B. NOT (p+e)/nb
      Real ndpdn = 0.0; // nb * (ni/nb)*dPi/dni = nb * dPi/dnb
      Real dpdT = 0.0; // dPi/dT
      Real dsdn = 0.0; // ni*dsi/dni - si = nb^2 * dSi/dnb
      Real dsdT = 0.0; // dsi/dT = nb * dSi/dT
      Real lt = log2_(T);

      int it;
      Real wt1;
      weight_idx_lt(&wt1, &it, lt);

      // 3D Tables
      for (int i=0; i<n_tables_3D; ++i) {
        Real ni, yi;
        GetPartialInputs3D(i, nb, Y, ni, yi);
        Real lni = log2_(ni);
        
        int in, iy;
        Real wn1, wy1;

        weight_idx_ln(i, &wn1, &in, lni);
        weight_idx_yi(i, &wy1, &iy, yi);

        h += exp2_(eval_at_inty(i, ECLOGP, in, it, iy, wn1, wt1, wy1)) - Pmin(i);
        h += exp2_(eval_at_inty(i, ECLOGE, in, it, iy, wn1, wt1, wy1));

        ndpdn += ni*eval_at_inty(i, ECDPDN, in, it, iy, wn1, wt1, wy1);
        dpdT += eval_at_inty(i, ECDPDT, in, it, iy, wn1, wt1, wy1);
        dsdn += ni*eval_at_inty(i, ECDSDN, in, it, iy, wn1, wt1, wy1) - eval_at_inty(i, ECENTD, in, it, iy, wn1, wt1, wy1);
        dsdT += eval_at_inty(i, ECDSDT, in, it, iy, wn1, wt1, wy1);
      }

      // 2D Tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real ni;
        GetPartialInputs2D(i, nb, Y, ni);
        Real lni = log2_(ni);
        
        int in;
        Real wn1;

        weight_idx_ln(i, &wn1, &in, lni);

        h += exp2_(eval_at_int(i, ECLOGP, in, it, wn1, wt1)) - Pmin(i);
        h += exp2_(eval_at_int(i, ECLOGE, in, it, wn1, wt1));

        ndpdn += ni*eval_at_int(i, ECDPDN, in, it, wn1, wt1);
        dpdT += eval_at_int(i, ECDPDT, in, it, wn1, wt1);
        dsdn += ni*eval_at_int(i, ECDSDN, in, it, wn1, wt1) - eval_at_int(i, ECENTD, in, it, wn1, wt1);
        dsdT += eval_at_int(i, ECDSDT, in, it, wn1, wt1);
      }

      // Photons
      if (use_photons) {
        h += (photonPressureConstant + photonEnergyConstant) * pow(T,4);

        dpdT += 4.0 * photonPressureConstant * pow(T,3.0);
        dsdn -= photonEntropyConstant * pow(T,3.0);
        dsdT += 3.0 * photonEntropyConstant * pow(T,2.0);
      }

      Real cs2 = (ndpdn - dpdT*dsdn/dsdT)/h;
      return Kokkos::sqrt(cs2);
    }

    /// Calculate the specific internal energy per unit mass
    KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(const Real nb, const Real T, const Real *Y) const {
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
    KOKKOS_INLINE_FUNCTION Real MinimumPressure(const Real nb, const Real *Y) const {
      return Pressure(nb, min_T, Y);
    }

    /// Get the maximum pressure at a given density and composition
    KOKKOS_INLINE_FUNCTION Real MaximumPressure(const Real nb, const Real *Y) const {
      return Pressure(nb, max_T, Y);
    }

    /// Get the minimum energy at a given density and composition
    KOKKOS_INLINE_FUNCTION Real MinimumEnergy(const Real nb, const Real *Y) const {
      return Energy(nb, min_T, Y);
    }

    /// Get the maximum energy at a given density and composition
    KOKKOS_INLINE_FUNCTION Real MaximumEnergy(const Real nb, const Real *Y) const {
      return Energy(nb, max_T, Y);
    }

    /// Temperature from energy density.
    KOKKOS_INLINE_FUNCTION Real TemperatureFromE(const Real nb, const Real e, const Real *Y) const {
      assert (initialised);
      return TemperatureFromVar<ECLOGE>(e, nb, Y);
    }

    /// Calculate the from pressure.
    KOKKOS_INLINE_FUNCTION Real TemperatureFromP(const Real nb, const Real p, const Real *Y) const {
      assert (initialised);
      Real p_target = p;
      for (int i=0; i<n_tables_3D+n_tables_2D; ++i) {
        p_target += Pmin(i);
      }
      return TemperatureFromVar<ECLOGP>(p_target, nb, Y);
    }

  protected:
    /// Low level functions not intended for outside use
    // Parse header line
    void ParseLine(std::string line, std::string& name, 
                   std::string& value, std::string& comment) const;
    // Read subtable files
    bool Read2DTableFromFile(std::string table_name, int table_idx);
    bool Read3DTableFromFile(std::string table_name, int table_idx);
    bool ReadTSharedTableFromFile(std::string table_name);

    // Temperature inversion
    // Template over which variable is being solved
    template<int iv>
    KOKKOS_INLINE_FUNCTION Real TemperatureFromVar(const Real var, const Real nb, const Real *Y) const {
      assert(initialised);

      // Indicies and weights for densities and compositions can be 
      // precalculated
      int in[MAX_TABLES], iy[MAX_TABLES];
      Real wn1[MAX_TABLES], wy1[MAX_TABLES];

      for (int i=0; i<n_tables_3D; ++i) {
        Real ni, yi;
        GetPartialInputs3D(i, nb, Y, ni, yi);

        weight_idx_ln(i, &(wn1[i]), &(in[i]), log2_(ni));
        weight_idx_yi(i, &(wy1[i]), &(iy[i]), yi);
      }

      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real ni;
        GetPartialInputs2D(i, nb, Y, ni);

        weight_idx_ln(i, &(wn1[i]), &(in[i]), log2_(ni));
      }
      
      auto f_idx = [=](int t_idx) {
        return RootFunctionIdx(t_idx, var, iv, in, iy, wn1, wy1, this);
      };

      int ilo = 0;
      int ihi = n_t_shared-1;

      Real flo = f_idx(ilo);
      Real fhi = f_idx(ihi);

      if (!(flo*fhi <= 0.0)) {
        // The root is not bounded by T_min and T_max, but not all hope
        // is lost

        if constexpr(iv==ECLOGE) {
          // Energy(T) should be monotonic, so we can check the signs of 
          // the function to see if the root is outside the bounds of the 
          // table.
          // f(idx) = (var_target - var(T_idx)) / var_target
          // if f(ilo) <= 0 then var_target < var(T=T_min), 
          // equally if f(ihi) >= 0 then var_target > var(T=T_max)
          if (flo<=0.0) {
            return t_shared(0);
          } else if (fhi>=0.0) {
            return t_shared(n_t_shared-1);
          }
        } else if constexpr(iv==ECLOGP) {
          // Pressure may not be monotonic, so first we sweep the whole
          // temperature axis to see if we can find some valid bounds
          while (!(flo*fhi <= 0)){
            if (ilo == ihi - 1) {
              // We swept the whole table and didn't find a root, now we
              // can check the edges to see if they imply the existence
              // of a root beyond the table
              ilo = 0;
              flo = f_idx(ilo);
              if (flo<=0.0) {
                return t_shared(0);
              } else if (fhi>=0.0) {
                return t_shared(n_t_shared-1);
              }
              break;
            } else {
              ilo += 1;
              flo = f_idx(ilo);
            }
          }
        }
      }
      
      // If we don't have a bounded root at this point then we complain
      if (!(flo*fhi <= 0)) {
        Real flo_ = f_idx(0);
        Real fhi_ = f_idx(n_t_shared-1);
        Kokkos::printf("Root not bound in TemperatureFromVar: nb=%e, Y[0]=%e\n", nb, Y[0]);
        Kokkos::printf("Root not bound in TemperatureFromVar: f(ilo)=%e, f(ihi)=%e\n", flo_, fhi_);
      }
      assert(flo*fhi <= 0);

      while (ihi - ilo > 1) {
        int ip = ilo + (ihi - ilo)/2;
        Real fp = f_idx(ip);
        if (fp*flo <= 0) {
          ihi = ip;
          fhi = fp;
        } else {
          ilo = ip;
          flo = fp;
        }
      }

      assert(ihi - ilo == 1);

      Real w_fp; // Solution to be calculated
      Real lb = 0.0; // Initial bounds for w.
      Real ub = 1.0;

      // calc exponential interpolation parameters
      Real lvar[MAX_TABLES+1], dlvar[MAX_TABLES+1]; // var_i(t) = exp(lvar[i] + w(t)*dlvar[i])
      
      // 3D Tables
      for (int i=0; i<n_tables_3D; ++i) {
        Real lvar_lb = (1.0-wn1[i]) * ((1.0-wy1[i]) * table(index3D(i, iv, in[i]+0, iy[i]+0, ilo))  +
                                            wy1[i]  * table(index3D(i, iv, in[i]+0, iy[i]+1, ilo))) +
                            wn1[i]  * ((1.0-wy1[i]) * table(index3D(i, iv, in[i]+1, iy[i]+0, ilo))  +
                                            wy1[i]  * table(index3D(i, iv, in[i]+1, iy[i]+1, ilo)));

        Real lvar_ub = (1.0-wn1[i]) * ((1.0-wy1[i]) * table(index3D(i, iv, in[i]+0, iy[i]+0, ihi))  +
                                            wy1[i]  * table(index3D(i, iv, in[i]+0, iy[i]+1, ihi))) +
                            wn1[i]  * ((1.0-wy1[i]) * table(index3D(i, iv, in[i]+1, iy[i]+0, ihi))  +
                                            wy1[i]  * table(index3D(i, iv, in[i]+1, iy[i]+1, ihi)));
        lvar[i] = lvar_lb;
        dlvar[i] = lvar_ub - lvar_lb;
      }

      // 2D Tables
      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real lvar_lb = (1.0-wn1[i]) * table(index2D(i, iv, in[i]+0, ilo)) +
                            wn1[i]  * table(index2D(i, iv, in[i]+1, ilo));

        Real lvar_ub = (1.0-wn1[i]) * table(index2D(i, iv, in[i]+0, ihi)) +
                            wn1[i]  * table(index2D(i, iv, in[i]+1, ihi));
              
        lvar[i] = lvar_lb;
        dlvar[i] = lvar_ub - lvar_lb;
      }

      // Radiation
      if (use_photons) {
        Real lvar_lb;
        Real lvar_ub;
        if constexpr(iv==ECLOGP) {
          lvar_lb = log2_(photonPressureConstant * pow(t_shared(ilo),4));
          lvar_ub = log2_(photonPressureConstant * pow(t_shared(ihi),4));
        } else if constexpr(iv==ECLOGE) {
          lvar_lb = log2_(photonEnergyConstant * pow(t_shared(ilo),4));
          lvar_ub = log2_(photonEnergyConstant * pow(t_shared(ihi),4));
        }
        lvar[n_tables_3D+n_tables_2D] = lvar_lb;
        dlvar[n_tables_3D+n_tables_2D] = lvar_ub - lvar_lb;
      }

      // As we switch from one method of calculating the root to another,
      // make sure the thing is still bounded
      Real flb = RootFunctionW(lb, var, iv, lvar, dlvar, this);
      Real fub = RootFunctionW(ub, var, iv, lvar, dlvar, this);
      if (!(flb*fub<=0.0)) {
        Real flo_ = f_idx(ilo);
        Real fhi_ = f_idx(ihi);
        Kokkos::printf("Root not bound in TemperatureFromVar: nb=%e, Y[0]=%e\n", nb, Y[0]);
        Kokkos::printf("Root not bound in TemperatureFromVar: f(%d)=%e, f(%d)=%e\n", ilo, flo_, ihi, fhi_);
        Kokkos::printf("Root not bound in TemperatureFromVar: f(%e)=%e, f(%e)=%e\n", lb, flb, ub, fub);
      }

      bool result = root.FalsePositionModified(RootFunctionW, lb, ub, w_fp, 1e-15, 1e-15, var, iv, lvar, dlvar, this);

      // Complain if root-solve was unsuccessful
      if (!result) {
        flb = RootFunctionW(lb, var, iv, lvar, dlvar, this);
        fub = RootFunctionW(ub, var, iv, lvar, dlvar, this);
        Kokkos::printf("Root not converged in FalsePositionModified: nb=%e, Y[0]=%e\n", nb, Y[0]);
        Kokkos::printf("Root not converged in FalsePositionModified: f(%e)=%e, f(%e)=%e\n", lb, flb, ub, fub);
      }
      assert(result);

      return exp2_(log_t_shared(ilo) + w_fp*dlog_t_shared);
    }

    // Accessing tables
    KOKKOS_INLINE_FUNCTION void GetPartialInputs3D(const int table_idx, const Real nb, const Real *Y, Real &ni, Real &yi) const {
      Real Ni, Yi;
      GetPartialNi(table_idx, Y, Ni);
      GetPartialYi(table_idx, Y, Yi);
      yi = Yi/Ni;
      ni = Ni*nb;
      return;
    }

    KOKKOS_INLINE_FUNCTION void GetPartialInputs2D(const int table_idx, const Real nb, const Real *Y, Real &ni) const {
      Real Ni;
      GetPartialNi(table_idx, Y,Ni);
      ni = Ni*nb;
      return;
    }

    KOKKOS_INLINE_FUNCTION void GetPartialNi(const int table_idx, const Real *Y, Real &Ni) const {
      Ni = n_weights(table_idx,0);
      for (int i=0; i<n_species; ++i) {
        Ni += n_weights(table_idx,1+i)*Y[i];
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION void GetPartialYi(const int table_idx, const Real *Y, Real &Yi) const {
      Yi = y_weights(table_idx,0);
      for (int i=0; i<n_species; ++i) {
        Yi += y_weights(table_idx,1+i)*Y[i];
      }
      return;
    }

    /* These have been factored out
    KOKKOS_INLINE_FUNCTION Real eval_at_nty(const int table_idx, const int vi, const Real ni, const Real T, const Real Yi) const {
      return eval_at_lnty(table_idx, vi, log2_(ni), log2_(T), Yi);
    }

    KOKKOS_INLINE_FUNCTION Real eval_at_nt(const int table_idx, const int vi, const Real ni, const Real T) const {
      return eval_at_lnt(table_idx, vi, log2_(ni), log2_(T));
    }

    KOKKOS_INLINE_FUNCTION Real eval_at_lnty(const int table_idx, const int iv, const Real ln, const Real lt, const Real yi) const {
      int in, iy, it;
      Real wn1, wy1, wt1;

      weight_idx_ln(table_idx, &wn1, &in, ln);
      weight_idx_yi(table_idx, &wy1, &iy, yi);
      weight_idx_lt(&wt1, &it, lt);

      return eval_at_inty(table_idx, iv, in, it, iy, wn1, wt1, wy1);
    }

    KOKKOS_INLINE_FUNCTION Real eval_at_lnt(const int table_idx, const int iv, const Real ln, const Real lt) const {
      int in, it;
      Real wn1, wt1;

      weight_idx_ln(table_idx, &wn1, &in, ln);
      weight_idx_lt(&wt1, &it, lt);

      return eval_at_int(table_idx, iv, in, it, wn1, wt1);
    }
    */

    KOKKOS_INLINE_FUNCTION Real eval_at_inty(const int table_idx, const int iv, const int in, const int it, const int iy, const Real wn1, const Real wt1, const Real wy1) const {
      return
        (1.0-wn1) * ((1.0-wy1) * ((1.0-wt1) * table(index3D(table_idx, iv, in+0, iy+0, it+0))   +
                                       wt1  * table(index3D(table_idx, iv, in+0, iy+0, it+1)))  +
                          wy1  * ((1.0-wt1) * table(index3D(table_idx, iv, in+0, iy+1, it+0))   +
                                       wt1  * table(index3D(table_idx, iv, in+0, iy+1, it+1)))) +
             wn1  * ((1.0-wy1) * ((1.0-wt1) * table(index3D(table_idx, iv, in+1, iy+0, it+0))   +
                                       wt1  * table(index3D(table_idx, iv, in+1, iy+0, it+1)))  +
                          wy1  * ((1.0-wt1) * table(index3D(table_idx, iv, in+1, iy+1, it+0))   +
                                       wt1  * table(index3D(table_idx, iv, in+1, iy+1, it+1))));
    }

    KOKKOS_INLINE_FUNCTION Real eval_at_int(const int table_idx, const int iv, const int in, const int it, const Real wn1, const Real wt1) const {
      return
        (1.0-wn1) * ((1.0-wt1) * table(index2D(table_idx, iv, in+0, it+0))   +
                          wt1  * table(index2D(table_idx, iv, in+0, it+1)))  +
             wn1  * ((1.0-wt1) * table(index2D(table_idx, iv, in+1, it+0))   +
                          wt1  * table(index2D(table_idx, iv, in+1, it+1)));
    }

    KOKKOS_INLINE_FUNCTION void weight_idx_ln(const int table_idx, Real *w1, int *in, const Real ln) const {
      int offset = offset_ni(table_idx);
      if (!(ln>log_ni(offset))) { // x <= xmin -> !(x > xmin)
        *in = 0;
        *w1 = 0.0;
      } else if (!(ln<log_ni(offset+nni(table_idx)-1))) { // x >= xmin -> !(x < xmax)
        *in = nni(table_idx)-2;
        *w1 = 1.0;
      } else {
        *in = (ln - log_ni(offset))*inv_dlog_ni(table_idx);
        *w1 = (ln - log_ni(offset + (*in)))*inv_dlog_ni(table_idx);
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION void weight_idx_yi(const int table_idx, Real *w1, int *iy, const Real yi) const {
      int offset = offset_yi(table_idx);
      if (!(yi>this->yi(offset))) {
        *iy = 0;
        *w1 = 0.0;
      } else if (!(yi<this->yi(offset+nyi(table_idx)-1))) {
        *iy = nyi(table_idx)-2;
        *w1 = 1.0;
      } else {
        *iy = (yi - this->yi(offset))*inv_dyi(table_idx);
        *w1 = (yi - this->yi(offset + (*iy)))*inv_dyi(table_idx);
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION void weight_idx_lt(Real *w1, int *it, const Real lt) const {
      if (!(lt>log_t_shared(0))) {
        *it = 0;
        *w1 = 0.0;
      } else if (!(lt<log_t_shared(n_t_shared - 1))) {
        *it = n_t_shared-2;
        *w1 = 1.0;
      } else{
        *it = (lt - log_t_shared(0))*inv_dlog_t_shared;
        *w1 = (lt - log_t_shared(*it))*inv_dlog_t_shared;
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION int index3D(const int table_idx, const int iv, const int in, const int iy, const int it) const {
      return offset_table(table_idx) + it + n_t_shared*(iy + nyi(table_idx)*(in + nni(table_idx)*iv));
    }

    KOKKOS_INLINE_FUNCTION int index2D(const int table_idx, const int iv, const int in, const int it) const {
      return offset_table(table_idx) + it + n_t_shared*(in + nni(table_idx)*iv);
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

    Real Pmin_fac; // relative value to offset pressure to ensure positive

    // Subtables
    int n_tables_3D;
    int n_tables_2D;
    int n_ni_full;
    int n_yi_full;
    int n_table_full;

    // Table storage
    DvceArray1D<int> nni, nyi;            // <number density, fraction> samples for each subtable
    DvceArray1D<Real> inv_dlog_ni, inv_dyi; // inverse <log number density, fraction> spacing for each subtable
    DvceArray1D<Real> Pmin;               // pressure offsets for where pressure<=0

    // Sequential table storage. 
    DvceArray1D<Real> ni, log_ni;                                   // <number density, log number density> for each subtable sequentially
    DvceArray1D<Real> yi;                                           // fractions for each subtable sequentially
    DvceArray1D<Real> table;                                        // data for each subtable sequentially
    DvceArray1D<int>  offset_ni, offset_yi, offset_table; // offsets for start of each subtables data

    DvceArray2D<int> y_weights, n_weights; // weights for calculating ni and yi from nb and Y

    // Shared temperature axis
    DvceArray1D<Real> t_shared, log_t_shared;
    int n_t_shared;
    Real dlog_t_shared;
    Real inv_dlog_t_shared;

    /// The root solvers.
    /// This calculates the root at a given temperature index
    class RootFunctorIdx {
      public:
        KOKKOS_INLINE_FUNCTION Real operator()(const int it, const Real var, 
                                               const int iv, const int *in, const int *iy, 
                                               const Real *wn1, const Real *wy1, 
                                               const EOSMultiTable* pparent) const {
          Real var_pt = 0.0;

          // 3D Tables
          for (int i=0; i<pparent->n_tables_3D; ++i) {
            var_pt += pparent->exp2_((1.0-wn1[i]) * ((1.0-wy1[i]) * pparent->table(pparent->index3D(i, iv, in[i]+0, iy[i]+0, it))  +
                                                          wy1[i]  * pparent->table(pparent->index3D(i, iv, in[i]+0, iy[i]+1, it))) +
                                          wn1[i]  * ((1.0-wy1[i]) * pparent->table(pparent->index3D(i, iv, in[i]+1, iy[i]+0, it))  +
                                                          wy1[i]  * pparent->table(pparent->index3D(i, iv, in[i]+1, iy[i]+1, it))));
          }

          // 2D Tables
          for (int i=pparent->n_tables_3D; i<pparent->n_tables_3D+pparent->n_tables_2D; ++i) {
            var_pt += pparent->exp2_((1.0-wn1[i]) * pparent->table(pparent->index2D(i, iv, in[i]+0, it)) +
                                          wn1[i]  * pparent->table(pparent->index2D(i, iv, in[i]+1, it)));
          }

          if (pparent->use_photons) {
            if (iv==ECLOGP) {
              var_pt += pparent->photonPressureConstant * pow(pparent->t_shared(it),4);
            } else if (iv==ECLOGE) {
              var_pt += pparent->photonEnergyConstant * pow(pparent->t_shared(it),4);
            }
          }

          return (var - var_pt)/var; // N.B error is expected to be relative
        }
    };

    /// This calculates the root at a given temperature weight. 
    /// Precomputed exponential interpolators to be passed in with lvar 
    /// and dlvar where var_i(t) = exp(lvar[i] + w(t)*dlvar[i]) on the 
    /// given interval.
    class RootFunctorW {
      public:
        KOKKOS_INLINE_FUNCTION Real operator()(const Real wt, const Real var, 
                                               const int iv, 
                                               const Real *lvar, const Real *dlvar, 
                                               const EOSMultiTable* pparent) const {
          Real var_pt = 0.0;

          // 3D Tables
          for (int i=0; i<pparent->n_tables_3D; ++i) {
            var_pt += pparent->exp2_(lvar[i] + wt*dlvar[i]);
          }

          // 2D Tables
          for (int i=pparent->n_tables_3D; i<pparent->n_tables_3D+pparent->n_tables_2D; ++i) {
            var_pt += pparent->exp2_(lvar[i] + wt*dlvar[i]);
          }

          if (pparent->use_photons) {
            int i = pparent->n_tables_3D+pparent->n_tables_2D;
            var_pt += pparent->exp2_(lvar[i] + wt*dlvar[i]);
          }

          return (var - var_pt)/var; // N.B error is expected to be relative
        }
    };

    NumTools::Root root;
    RootFunctorIdx RootFunctionIdx;
    RootFunctorW RootFunctionW;

}; // class EOSMultiTable

}; // namespace Primitive


#endif // EOS_PRIMITIVE_SOLVER_EOS_MULTITABLE_HPP_