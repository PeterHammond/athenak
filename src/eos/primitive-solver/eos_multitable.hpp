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
    KOKKOS_INLINE_FUNCTION Real Pressure(const Real nb, const Real T, const Real *Y) const {
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
    KOKKOS_INLINE_FUNCTION Real Entropy(const Real nb, const Real T, const Real *Y) const {
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
    KOKKOS_INLINE_FUNCTION Real Enthalpy(const Real nb, const Real T, const Real *Y) const {
      Real const P = Pressure(nb, T, Y);
      Real const e = Energy(nb, T, Y);
      return (P + e)/nb;
    }

    /// Calculate the sound speed.
    KOKKOS_INLINE_FUNCTION Real SoundSpeed(const Real nb, const Real T, const Real *Y) const {
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

      Real cs2 = (dpdn - dpdT*dsdn/(nb*dsdT))/h;
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

    /// Temperature from energy density
    KOKKOS_INLINE_FUNCTION Real TemperatureFromE(const Real nb, const Real e, const Real *Y) const {
      assert (initialised);
      Real e_min = MinimumEnergy(nb, Y);
      Real e_max = MaximumEnergy(nb, Y);
      if (e <= e_min) {
        return min_T;
      } else if (e >= e_max) {
        return max_T;
      } else {
        return TemperatureFromVar(ECLOGE, e, nb, Y);
      }
    }

    /// Calculate the temperature using.
    KOKKOS_INLINE_FUNCTION Real TemperatureFromP(const Real nb, const Real p, const Real *Y) const {
      assert (initialised);
      Real p_min = MinimumPressure(nb, Y);
      Real p_max = MaximumPressure(nb, Y);
      if (p <= p_min) {
        return min_T;
      } else if (p >= p_max) {
        return max_T;
      } else {
        Real p_target = p;
        
        for (int i=0; i<n_tables_3D+n_tables_2D; ++i) {
          p_target += Pmin(i);
        }
        return TemperatureFromVar(ECLOGP, p_target, nb, Y);
      }
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


    // Evaluation of subtables
    KOKKOS_INLINE_FUNCTION Real PartialPressure3D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return exp2_(eval_at_nty(table_idx, ECLOGP, ni, T, yi)) - Pmin(table_idx);
    }

    KOKKOS_INLINE_FUNCTION Real PartialPressure2D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return exp2_(eval_at_nt(table_idx, ECLOGP, ni, T)) - Pmin(table_idx);
    }

    KOKKOS_INLINE_FUNCTION Real PartialEnergyDensity3D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return exp2_(eval_at_nty(table_idx, ECLOGE, ni, T, yi));
    }

    KOKKOS_INLINE_FUNCTION Real PartialEnergyDensity2D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return exp2_(eval_at_nt(table_idx, ECLOGE, ni, T));
    }

    KOKKOS_INLINE_FUNCTION Real PartialEntropyDensity3D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return eval_at_nty(table_idx, ECENTD, ni, T, yi);
    }

    KOKKOS_INLINE_FUNCTION Real PartialEntropyDensity2D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return eval_at_nt(table_idx, ECENTD, ni, T);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDPDN3D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return (ni/nb)*eval_at_nty(table_idx, ECDPDN, ni, T, yi);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDPDN2D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return (ni/nb)*eval_at_nt(table_idx, ECDPDN, ni, T);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDPDT3D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return eval_at_nty(table_idx, ECDPDT, ni, T, yi);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDPDT2D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return eval_at_nt(table_idx, ECDPDT, ni, T);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDSDN3D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      Real s    = eval_at_nty(table_idx, ECENTD, ni, T, yi);
      Real dsdn = eval_at_nty(table_idx, ECDSDN, ni, T, yi);
      return ni*dsdn - s;
    }

    KOKKOS_INLINE_FUNCTION Real PartialDSDN2D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      Real s    = eval_at_nt(table_idx, ECENTD, ni, T);
      Real dsdn = eval_at_nt(table_idx, ECDSDN, ni, T);
      return ni*dsdn - s;
    }

    KOKKOS_INLINE_FUNCTION Real PartialDSDT3D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni, yi;
      GetPartialInputs3D(table_idx, nb, Y, ni, yi);
      return eval_at_nty(table_idx, ECDSDT, ni, T, yi);
    }

    KOKKOS_INLINE_FUNCTION Real PartialDSDT2D(const int table_idx, const Real nb, const Real T, const Real *Y) const {
      Real ni;
      GetPartialInputs2D(table_idx, nb, Y, ni);
      return eval_at_nt(table_idx, ECDSDT, ni, T);
    }

    // Temperature inversion
    KOKKOS_INLINE_FUNCTION Real TemperatureFromVar(const int iv, const Real var, const Real nb, const Real *Y) const {
      assert(initialised);
      assert(iv==ECLOGP || iv==ECLOGE);

      // Indicies and weights
      int in[MAX_TABLES], iy[MAX_TABLES];
      Real wn1[MAX_TABLES], wy1[MAX_TABLES];

      for (int i=0; i<n_tables_3D; ++i) {
        Real ni, yi;
        GetPartialInputs3D(i, nb, Y, ni, yi);

        weight_idx_ln(i, &(wn1[i]), &(in[i]), log2_(ni));
        weight_idx_yi(i, &(wy1[i]), &(iy[i]), yi);

        // printf("%d %e %e %e %e\n",i,nb,Y[0],ni,yi);
      }

      for (int i=n_tables_3D; i<n_tables_3D+n_tables_2D; ++i) {
        Real ni;
        GetPartialInputs2D(i, nb, Y, ni);

        weight_idx_ln(i, &(wn1[i]), &(in[i]), log2_(ni));

        // printf("%d %e %e %e\n",i,nb,Y[0],ni);
      }
      
      // TODO Fix
      auto f_idx = [=](int t_idx) {
        return RootFunctionIdx(t_idx, var, iv, in, iy, wn1, wy1, this);
      };

      int ilo = 0;
      int ihi = n_t_shared-1;

      Real flo = f_idx(ilo);
      Real fhi = f_idx(ihi);
      
      /*
      Real var_lo, var_hi;
      if (iv==0) {
        var_lo = log2_(Pressure(nb, t_union(ilo), Y));
        var_hi = log2_(Pressure(nb, t_union(ihi), Y));
      } else if (iv==1) {
        var_lo = log2_(Energy(nb, t_union(ilo), Y));
        var_hi = log2_(Energy(nb, t_union(ihi), Y));
      }

      printf("%d %e %e %e %e %e %d %d %e %e %e %e\n",iv,var,var_lo,var_hi,nb,Y[0],ilo,ihi,lt_lo,lt_hi,flo,fhi);
      */

      while (flo*fhi>0){
        if (ilo == ihi - 1) {
          // if (abs(fhi) < abs(flo)) {
          //   return exp2_(lt_hi);
          // }
          break;
        } else {
          ilo += 1;
          flo = f_idx(ilo);
        }
      }

      assert(flo*fhi <= 0);
      if (!(flo*fhi <= 0)) {
        Real flo_ = f_idx(0);
        Real fhi_ = f_idx(n_t_shared-1);
        Kokkos::printf("Root not bounded in TemperatureFromVar: f(ilo)=%e, f(ihi)=%e\n", flo_, fhi_);
      }
      assert(flo*fhi <= 0);

      // printf("- %d %d %e %e %e %e\n",ilo,ihi,log_t_union(ilo),log_t_union(ihi),f(log_t_union(ilo)),f(log_t_union(ihi)));

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

      // printf("- %d %d %e %e %e %e\n",ilo,ihi,log_t_union(ilo),log_t_union(ihi),f(log_t_union(ilo)),f(log_t_union(ihi)));

      assert(ihi - ilo == 1);

      Real w_fp; // Solution to be calculated
      Real lb = 0.0; // Initial bounds for w.
      Real ub = 1.0;

      // calc exponential interpolation parameters
      Real lvar[MAX_TABLES], dlvar[MAX_TABLES]; // var_i(t) = exp(lvar[i] + w(t)*dlvar[i])
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

      bool result = root.FalsePositionModified(RootFunctionW, lb, ub, w_fp, 1e-15, 1e-15, var, iv, ilo, lvar, dlvar, this);
      if (!result) {
        Real flb = RootFunctionW(lb, var, iv, ilo, lvar, dlvar, this);
        Real fub = RootFunctionW(ub, var, iv, ilo, lvar, dlvar, this);
        Kokkos::printf("Root not converged in FalsePositionModified: nb=%e, Y[0]=%e\n", nb, Y[0]);
        Kokkos::printf("Root not converged in FalsePositionModified: f(%e)=%e, f(%e)=%e\n", lb, flb, ub, fub);
      }
      assert(result);
      
      // printf("- %e %e %e %e %e %e\n",lb,ub,lt_fp,lt_fp - log_t_offset,exp2_(lt_fp - log_t_offset),f(lt_fp - log_t_offset));
      /*
      Real var_Tinit;
      if (iv==ECLOGP) {
        Real p_min_offset = 0.0;
        var_Tinit = log2_(Pressure(nb, 0.2, Y));
      } else if (iv==ECLOGE) {
        var_Tinit = log2_(Energy(nb, 0.2, Y));
      }

      Real var_Tsolve;
      Real Tsolve = exp2_(lt_fp - log_t_offset);
      if (iv==ECLOGP) {
        var_Tsolve = log2_(Pressure(nb, Tsolve, Y));
      } else if (iv==ECLOGE) {
        var_Tsolve = log2_(Energy(nb, Tsolve, Y));
      }

      printf("%d %e %e %e %e %e %e %d\n", iv, var, var_Tsolve, var_Tinit, nb, Y[0], exp2_(lt_fp - log_t_offset), result);
      */

      return exp2_(log_t_shared(ilo) + w_fp*dlog_t_shared);
    }

    // Accessing tables
    KOKKOS_INLINE_FUNCTION void GetPartialInputs3D(const int table_idx, const Real nb, const Real *Y, Real &ni, Real &yi) const {
      Real Ni, Yi;
      GetPartialNi(table_idx, Y, Ni);
      GetPartialYi(table_idx, Y, Yi);
      yi = Yi/Ni;
      ni = Ni*nb;
      // yi = Kokkos::max(Kokkos::min(yi,this->yi(offset_yi(table_idx)+nyi(table_idx)-1)),this->yi(offset_yi(table_idx)));
      // ni = Kokkos::max(Kokkos::min(ni,this->ni(offset_ni(table_idx)+nni(table_idx)-1)),this->ni(offset_ni(table_idx)));
      return;
    }

    KOKKOS_INLINE_FUNCTION void GetPartialInputs2D(const int table_idx, const Real nb, const Real *Y, Real &ni) const {
      Real Ni;
      GetPartialNi(table_idx, Y,Ni);
      ni = Ni*nb;
      // ni = Kokkos::max(Kokkos::min(ni,this->ni(offset_ni(table_idx)+nni(table_idx)-1)),this->ni(offset_ni(table_idx)));
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
      weight_idx_lt(table_idx, &wt1, &it, lt);

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

    KOKKOS_INLINE_FUNCTION Real eval_at_lnt(const int table_idx, const int iv, const Real ln, const Real lt) const {
      int in, it;
      Real wn1, wt1;

      weight_idx_ln(table_idx, &wn1, &in, ln);
      weight_idx_lt(table_idx, &wt1, &it, lt);

      return
        (1.0-wn1) * ((1.0-wt1) * table(index2D(table_idx, iv, in+0, it+0))   +
                          wt1  * table(index2D(table_idx, iv, in+0, it+1)))  +
             wn1  * ((1.0-wt1) * table(index2D(table_idx, iv, in+1, it+0))   +
                          wt1  * table(index2D(table_idx, iv, in+1, it+1)));
    }

    KOKKOS_INLINE_FUNCTION void weight_idx_ln(const int table_idx, Real *w1, int *in, const Real ln) const {
      int offset = offset_ni(table_idx);
      if (ln<=log_ni(offset)) {
        *in = 0;
        *w1 = 0.0;
      } else if (ln>=log_ni(offset+nni(table_idx)-1)) {
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
      if (yi<=this->yi(offset)) {
        *iy = 0;
        *w1 = 0.0;
      } else if (yi>=this->yi(offset+nyi(table_idx)-1)) {
        *iy = nyi(table_idx)-2;
        *w1 = 1.0;
      } else {
        *iy = (yi - this->yi(offset))*inv_dyi(table_idx);
        *w1 = (yi - this->yi(offset + (*iy)))*inv_dyi(table_idx);
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION void weight_idx_lt(const int table_idx, Real *w1, int *it, const Real lt) const {
      if (lt<=log_t_shared(0)) {
        *it = 0;
        *w1 = 0.0;
      } else if (lt>=log_t_shared(n_t_shared - 1)) {
        *it = n_t_shared-2;
        *w1 = 1.0;
      } else{
        *it = (lt - log_t_shared(0))*inv_dlog_t_shared;
        *w1 = (lt - log_t_shared(*it))*inv_dlog_t_shared;
      }
      return;
    }

    KOKKOS_INLINE_FUNCTION int index3D(const int table_idx, const int iv, const int in, const int iy, const int it) const {
      // return offset_table(table_idx) + iv*(nni(table_idx)*nyi(table_idx)*nt(table_idx)) + in*(nyi(table_idx)*nt(table_idx)) + iy*(nt(table_idx)) + it;
      return offset_table(table_idx) + it + n_t_shared*(iy + nyi(table_idx)*(in + nni(table_idx)*iv));
    }

    KOKKOS_INLINE_FUNCTION int index2D(const int table_idx, const int iv, const int in, const int it) const {
      // return offset_table(table_idx) + iv*(nni(table_idx)*nt(table_idx)) + in*(nt(table_idx)) + it;
      return offset_table(table_idx) + it + n_t_shared*(in + nni(table_idx)*iv);
    }

    /*
    KOKKOS_INLINE_FUNCTION void test_temperature_recovery(Real nb, Real T, Real *Y) const {
      Real pressure = Pressure(nb, T, Y);
      Real energy   = Energy(nb, T, Y);

      Real T_p = TemperatureFromP(nb, pressure, Y);
      Real T_e = TemperatureFromE(nb, energy, Y);

      Real p_new = Pressure(nb, T_p, Y);
      Real e_new = Energy(nb, T_e, Y);

      Real T_err_p = T_p - T;
      Real T_err_e = T_e - T;

      Real p_err = p_new - pressure;
      Real e_err = e_new - energy;

      printf("Temperature recovery test at: nb=%e, T=%e, Ye=%f\n", nb, T, Y[0]);
      printf("TemperatureFromP: p=%e, T=%e, p(T)=%e\n", pressure, T_p, p_new);
      printf("abserr(T)=%e, relerr(T)=%e, abserr(p)=%e, relerr(p)=%e\n", T_err_p, T_err_p/T, p_err, p_err/pressure);
      printf("TemperatureFromE: e=%e, T=%e, e(T)=%e\n", energy, T_e, e_new);
      printf("abserr(T)=%e, relerr(T)=%e, abserr(e)=%e, relerr(e)=%e\n", T_err_e, T_err_e/T, e_err, e_err/energy);

      return;
    }
    */

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

    // TODO Fix
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
                                               const int iv, const int it, 
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
            Real t_current = pparent->exp2_(pparent->log_t_shared(it) + wt*pparent->dlog_t_shared);
            if (iv==ECLOGP) {
              var_pt += pparent->photonPressureConstant * pow(t_current,4);
            } else if (iv==ECLOGE) {
              var_pt += pparent->photonEnergyConstant * pow(t_current,4);
            }
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