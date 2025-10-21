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
#include <vector>

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
    printf("Reading MultiTable header file: %s\n", fname.c_str());
    std::string line, name, value, comment;
    int nline = -1;
    int err = -1;
    int min_Y_start = -1;
    std::vector<std::string> fnames_3D, fnames_2D;
    std::string T_union_fname;
    int y_weights_start = -1, n_weights_start = -1;
    int* y_weights_;
    int *n_weights_;
    while (std::getline(file, line)) {
      nline++;

      // Sanitise tabs
      if (line.find('\t') != std::string::npos) {
        line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
      }
      if (line.empty()) continue;                           // skip blank line
      std::size_t first_char = line.find_first_not_of(" "); // skip white space
      if (first_char == std::string::npos) continue;        // line is all white space
      if (line.compare(first_char, 1, "#") == 0) continue;  // skip comments

      ParseLine(line, name, value, comment);
      // printf("line: %d, name: %s, value: %s, comment: %s\n", nline, name.c_str(), value.c_str(), comment.c_str());


      if (name.compare(0,11,"n_3D_tables") == 0) {n_tables_3D = std::stoi(value); continue;}
      if (name.compare(0,11,"n_2D_tables") == 0) {n_tables_2D = std::stoi(value); continue;}
      if (name.compare(0,9,"n_species") == 0) {n_species = std::stoi(value); continue;}
      if (name.compare(0,4,"n_ni") == 0) {n_ni_full = std::stoi(value); continue;}
      if (name.compare(0,4,"n_yi") == 0) {n_yi_full = std::stoi(value); continue;}
      if (name.compare(0,7,"n_T_tab") == 0) {n_T_full = std::stoi(value); continue;}
      if (name.compare(0,9,"n_T_union") == 0) {n_T_union = std::stoi(value); continue;}
      if (name.compare(0,7,"n_table") == 0) {n_table_full = std::stoi(value); continue;}
      if (name.compare(0,2,"mb") == 0) {mb = std::stod(value); continue;}
      if (name.compare(0,5,"min_n") == 0) {min_n = std::stod(value); continue;}
      if (name.compare(0,5,"max_n") == 0) {max_n = std::stod(value); continue;}
      if (name.compare(0,5,"min_T") == 0) {min_T = std::stod(value); continue;}
      if (name.compare(0,5,"max_T") == 0) {max_T = std::stod(value); continue;}
      if (name.compare(0,5,"min_Y") == 0) {
        if (min_Y_start == -1) {min_Y_start = nline;}
        min_Y[(nline - min_Y_start)/2] = std::stod(value); 
        continue;
      }
      if (name.compare(0,5,"max_Y") == 0) {
        max_Y[(nline - 1 - min_Y_start)/2] = std::stod(value); 
        continue;
      }
      if (name.compare(0,9,"table_3D_") == 0) {fnames_3D.push_back(value); continue;}
      if (name.compare(0,9,"table_2D_") == 0) {fnames_2D.push_back(value); continue;}
      if (name.compare(0,7,"T_union") == 0) {T_union_fname = value; continue;}
      if (name.compare(0,4,"wY_(") == 0) {
        if (y_weights_start == -1) {
          y_weights_start = nline;
          y_weights_ = new int[n_tables_3D*(1+n_species)];
        }
        y_weights_[nline - y_weights_start] = std::stoi(value);
        continue;
      }
      if (name.compare(0,4,"wN_(") == 0) {
        if (n_weights_start == -1) {
          n_weights_start = nline;
          n_weights_ = new int[(n_tables_3D+n_tables_2D)*(1+n_species)];
        }
        n_weights_[nline - n_weights_start] = std::stoi(value);
        continue;
      }

      err = nline; // We read some data and have no idea what to do with it.
      if (err >= 0) {
        break;
      }

    }
    file.close(); // Close header.

    if (n_tables_3D+n_tables_2D > MAX_TABLES) {
      printf("Number of tables exceeds MAX_TABLES=%d, increase and recompile.\n",MAX_TABLES);
      abort();
      return;
    }

    if (err>=0) {
      printf("error reading MultiTable header file: %d\n",err);
      abort();
      return;
    } else {
      printf("n_tables_3D:  %d\n",n_tables_3D);
      printf("n_tables_2D:  %d\n",n_tables_2D);
      printf("n_species:    %d\n",n_species);
      printf("n_ni_full:    %d\n",n_ni_full);
      printf("n_yi_full:    %d\n",n_yi_full);
      printf("n_T_full:     %d\n",n_T_full);
      printf("n_T_union:    %d\n",n_T_union);
      printf("n_table_full: %d\n",n_table_full);
      printf("mb:           %e\n",mb);
      printf("min_n:        %e\n",min_n);
      printf("max_n:        %e\n",max_n);
      printf("min_T:        %e\n",min_T);
      printf("max_T:        %e\n",max_T);
      for (int idx=0; idx<n_species; ++idx) {
        printf("min_Y[%d]:     %e\n",idx,min_Y[idx]);
        printf("max_Y[%d]:     %e\n",idx,max_Y[idx]);
      }
      for (int idx=0; idx<n_tables_3D; ++idx) {
        printf("table_3D[%d]:  %s\n", idx, fnames_3D[idx].c_str());
      }
      for (int idx=0; idx<n_tables_2D; ++idx) {
        printf("table_2D[%d]:  %s\n", idx, fnames_2D[idx].c_str());
      }
      printf("T_union:      %s\n",T_union_fname.c_str());
      for (int idx=0; idx<n_tables_3D; ++idx) {
        for (int jdx=0; jdx<1+n_species; ++jdx) {
          printf("wY[%d,%d]:      %d\n",idx,jdx,y_weights_[idx*(1+n_species) + jdx]);
        }
      }
      for (int idx=0; idx<n_tables_3D+n_tables_2D; ++idx) {
        for (int jdx=0; jdx<1+n_species; ++jdx) {
          printf("wN[%d,%d]:      %d\n",idx,jdx,n_weights_[idx*(1+n_species) + jdx]);
        }
      }
    }

    /// Resize all device arrays now we have the info we need
    // <number density, fraction, temperature> samples for each subtable
    Kokkos::realloc(nni, n_tables_3D+n_tables_2D);
    Kokkos::realloc(nyi, n_tables_3D);
    Kokkos::realloc(nt,  n_tables_3D+n_tables_2D);

    Kokkos::realloc(inv_log_ni, n_tables_3D+n_tables_2D);
    Kokkos::realloc(inv_yi,     n_tables_3D);
    Kokkos::realloc(inv_log_t,  n_tables_3D+n_tables_2D);

    Kokkos::realloc(Pmin, n_tables_3D+n_tables_2D);

    // offsets for start of each subtables data
    Kokkos::realloc(offset_ni,    n_tables_3D+n_tables_2D);
    Kokkos::realloc(offset_yi,    n_tables_3D);
    Kokkos::realloc(offset_t,     n_tables_3D+n_tables_2D);
    Kokkos::realloc(offset_table, n_tables_3D+n_tables_2D);

    // <number density, log number density> for each subtable sequentially
    Kokkos::realloc(ni,     n_ni_full);
    Kokkos::realloc(log_ni, n_ni_full);

    // fractions for each subtable sequentially
    Kokkos::realloc(yi, n_yi_full);

    // <temperature, log temperature> for each subtable sequentially
    Kokkos::realloc(t,     n_T_full);
    Kokkos::realloc(log_t, n_T_full);

    // Union of temperatures for rootsolver
    Kokkos::realloc(t_union,     n_T_union);
    Kokkos::realloc(log_t_union, n_T_union);

    // data for each subtable sequentially
    Kokkos::realloc(table, ECNVARS*n_table_full);

    // weights for calculating ni and yi from nb and Y
    Kokkos::realloc(y_weights, n_tables_3D,             1+n_species);
    Kokkos::realloc(n_weights, n_tables_3D+n_tables_2D, 1+n_species);

    // Weights are read from header, so we can update now
    HostArray2D<int>::HostMirror host_y_weights = create_mirror_view(y_weights);
    HostArray2D<int>::HostMirror host_n_weights = create_mirror_view(n_weights);

    for (int idx=0; idx<n_tables_3D; ++idx) {
      for (int jdx=0; jdx<1+n_species; ++jdx) {
        host_y_weights(idx,jdx) = y_weights_[idx*(1+n_species) + jdx];
      }
    }

    for (int idx=0; idx<n_tables_3D+n_tables_2D; ++idx) {
      for (int jdx=0; jdx<1+n_species; ++jdx) {
        host_n_weights(idx,jdx) = n_weights_[idx*(1+n_species) + jdx];
      }
    }

    Kokkos::deep_copy(y_weights, host_y_weights);
    Kokkos::deep_copy(n_weights, host_n_weights);



    // Minimum enthalpy will be updated by each subtable
    min_h = 0.0;

    // Read subtables
    for (int idx=0; idx<n_tables_3D; ++idx) {
      bool read_success = Read3DTableFromFile(fnames_3D[idx],idx);
      if (!read_success) {
        printf("error reading MultiTable 3D subtable: %s\n",fnames_3D[idx].c_str());
        err = 100+idx;
      }
    }

    for (int idx=0; idx<n_tables_2D; ++idx) {
      bool read_success = Read2DTableFromFile(fnames_2D[idx],n_tables_3D+idx);
      if (!read_success) {
        printf("error reading MultiTable 2D subtable: %s\n",fnames_2D[idx].c_str());
        err = 100+n_tables_3D+idx;
      }
    }

    bool read_success = ReadTUnionTableFromFile(T_union_fname);
    if (!read_success) {
    printf("error reading MultiTable T Union subtable.\n");
      err = 200;
    }

    if (err<0) {
      initialised = true;

      for (int idx=0; idx<n_tables_3D+n_tables_2D; ++idx) {
        printf("ni offset %d: %d\n",idx,offset_ni(idx));
      }

      for (int idx=0; idx<n_tables_3D; ++idx) {
        printf("yi offset %d: %d\n",idx,offset_yi(idx));
      }

      for (int idx=0; idx<n_tables_3D+n_tables_2D; ++idx) {
        printf("t offset %d: %d\n",idx,offset_t(idx));
      }

      for (int idx=0; idx<n_tables_3D+n_tables_2D; ++idx) {
        printf("table offset %d: %d\n",idx,offset_table(idx));
      }
    } else {
      printf("Read err: %d\n", err);
    }
  }


  Real test_nb = 0.16161616;
  Real test_T  = 0.22222222;
  Real* test_Ye = new Real[MAX_SPECIES];
  test_Ye[0] = 0.0555555555;
  test_temperature_recovery(test_nb,test_T,test_Ye);

  return;
}


template<typename LogPolicy>
void EOSMultiTable<LogPolicy>::ParseLine(std::string line, std::string& name, 
                                         std::string& value, std::string& comment) const {
std::size_t first_char, last_char, equal_char, hash_char, len;
  first_char = line.find_first_not_of(" ");   // find first non-white space
  equal_char = line.find_first_of("=");       // find "=" char
  hash_char  = line.find_first_of("#");       // find "#" (optional)

  // copy substring into name, remove white space at end of name
  len = equal_char - first_char;
  name.assign(line, first_char, len);

  last_char = name.find_last_not_of(" ");
  name.erase(last_char+1, std::string::npos);

  // copy substring into value, remove white space at start and end
  len = hash_char - equal_char - 1;
  value.assign(line, equal_char+1, len);

  first_char = value.find_first_not_of(" ");
  value.erase(0, first_char);

  last_char = value.find_last_not_of(" ");
  value.erase(last_char+1, std::string::npos);

  // copy substring into comment, if present
  if (hash_char != std::string::npos) {
    comment = line.substr(hash_char);
  } else {
    comment = "";
  }
}

template<typename LogPolicy>
bool EOSMultiTable<LogPolicy>::Read3DTableFromFile(std::string fname, int table_idx) {
  bool success = true;
  /// Create host mirrors of device arrays to read into, and copy
  HostArray1D<int>::HostMirror  host_nni = create_mirror_view(nni), host_nyi = create_mirror_view(nyi), host_nt = create_mirror_view(nt);
  HostArray1D<Real>::HostMirror host_inv_log_ni = create_mirror_view(inv_log_ni), host_inv_yi = create_mirror_view(inv_yi), host_inv_log_t = create_mirror_view(inv_log_t);
  HostArray1D<Real>::HostMirror host_Pmin = create_mirror_view(Pmin);
  HostArray1D<int>::HostMirror  host_offset_ni = create_mirror_view(offset_ni), host_offset_yi = create_mirror_view(offset_yi), host_offset_t = create_mirror_view(offset_t), host_offset_table = create_mirror_view(offset_table);
  HostArray1D<Real>::HostMirror host_ni = create_mirror_view(ni), host_log_ni = create_mirror_view(log_ni);
  HostArray1D<Real>::HostMirror host_yi = create_mirror_view(yi);
  HostArray1D<Real>::HostMirror host_t = create_mirror_view(t), host_log_t = create_mirror_view(log_t);
  HostArray1D<Real>::HostMirror host_table = create_mirror_view(table);

  // Copy data from device to host
  Kokkos::deep_copy(host_nni, nni);
  Kokkos::deep_copy(host_nyi, nyi);
  Kokkos::deep_copy(host_nt, nt);
  Kokkos::deep_copy(host_inv_log_ni, inv_log_ni);
  Kokkos::deep_copy(host_inv_yi, inv_yi);
  Kokkos::deep_copy(host_inv_log_t, inv_log_t);
  Kokkos::deep_copy(host_Pmin, Pmin);
  Kokkos::deep_copy(host_offset_ni, offset_ni);
  Kokkos::deep_copy(host_offset_yi, offset_yi);
  Kokkos::deep_copy(host_offset_t, offset_t);
  Kokkos::deep_copy(host_offset_table, offset_table);
  Kokkos::deep_copy(host_ni, ni);
  Kokkos::deep_copy(host_log_ni, log_ni);
  Kokkos::deep_copy(host_yi, yi);
  Kokkos::deep_copy(host_t, t);
  Kokkos::deep_copy(host_log_t, log_t);

  // Read subtable
  TableReader::Table subtable;
  auto read_result = subtable.ReadTable(fname);
  if (read_result.error != TableReader::ReadResult::SUCCESS) {
    printf("error opening MultiTable 3D subtable: %s\n",fname.c_str());
    success = false;
    return success;
  }
  // Make sure table has correct dimentions
  if (subtable.GetNDimensions()!=3) {
    printf("MultiTable 3D subtable does not match expected number of dimensions: %ld\n",subtable.GetNDimensions());
    success = false;
    return success;
  }

  // Read scalars
  //printf("Read scalars.\n");
  auto& table_scalars = subtable.GetScalars();
  min_h += table_scalars.at("h_min");

  // Read dims
  //printf("Read dims.\n");
  auto& point_info = subtable.GetPointInfo();
  host_nni(table_idx) = point_info[0].second;
  host_nyi(table_idx) = point_info[1].second;
  host_nt(table_idx)  = point_info[2].second;

  if (table_idx==0) {
    host_offset_ni(table_idx) = 0;
    host_offset_yi(table_idx) = 0;
    host_offset_t(table_idx) = 0;
    host_offset_table(table_idx) = 0;
  }

  if (table_idx!=(n_tables_3D+n_tables_2D-1)){
    host_offset_ni(table_idx+1) = host_offset_ni(table_idx) + host_nni(table_idx);
    if (table_idx!=(n_tables_3D-1)) {
      host_offset_yi(table_idx+1) = host_offset_yi(table_idx) + host_nyi(table_idx);
    }
    host_offset_t(table_idx+1) = host_offset_t(table_idx) + host_nt(table_idx);
    host_offset_table(table_idx+1) = host_offset_table(table_idx) + host_nni(table_idx)*host_nyi(table_idx)*host_nt(table_idx)*ECNVARS;
  }
  

  //printf("Read nb.\n");
  { // read nb
    Real * table_ni = subtable["ni"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      host_ni(host_offset_ni(table_idx)+idx_ni)     = table_ni[idx_ni];
      host_log_ni(host_offset_ni(table_idx)+idx_ni) = log2_(host_ni(host_offset_ni(table_idx)+idx_ni));
    }
    host_inv_log_ni(table_idx) = 1.0/(host_log_ni(host_offset_ni(table_idx)+1) - host_log_ni(host_offset_ni(table_idx)+0));
  }

  //printf("Read yq.\n");
  { // read yq
    Real * table_yi = subtable["yi"];
    for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
      host_yi(host_offset_yi(table_idx)+idx_yi) = table_yi[idx_yi];
    }
    host_inv_yi(table_idx) = 1.0/(host_yi(host_offset_yi(table_idx)+1) - host_yi(host_offset_yi(table_idx)+0));
  }

  //printf("Read T.\n");
  { // read T
    Real * table_t = subtable["t"];
    for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
      host_t(host_offset_t(table_idx)+idx_t)     = table_t[idx_t];
      host_log_t(host_offset_t(table_idx)+idx_t) = log2_(host_t(host_offset_t(table_idx)+idx_t));
    }
    
    host_inv_log_t(table_idx) = 1.0/(host_log_t(host_offset_t(table_idx)+1) - host_log_t(host_offset_t(table_idx)+0));
  }

  //printf("Read pressure.\n");
  { // Read pressure
    Real * table_press = subtable["pressure"];
    Real Pmin_read = table_press[0];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
        for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
          size_t idx_flat_input = idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*idx_ni);
          Real p_current = table_press[idx_flat_input];
          if (p_current < Pmin_read) {Pmin_read = p_current;}
        }
      }
    }

    if (Pmin_read<0.0) {
      Pmin(table_idx) = -Pmin_read * (1.0 + Pmin_fac);
    } else {
      Pmin(table_idx) = 0.0;
    }

    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
        for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
          size_t idx_flat_input = idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*idx_ni);
          size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*(idx_ni + host_nni(table_idx)*ECLOGP));
          Real p_current = table_press[idx_flat_input];
          host_table(idx_flat_table) = log2_(p_current + Pmin(table_idx));
        }
      }
    }
  }

  //printf("Read energy.\n");
  { // Read energy
    Real * table_energy = subtable["energy"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
        for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
          size_t idx_flat_input = idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*idx_ni);
          size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*(idx_ni + host_nni(table_idx)*ECLOGE));
          Real e_current = table_energy[idx_flat_input];
          host_table(idx_flat_table) = log2_(e_current);
        }
      }
    }
  }

  //printf("Read entropy.\n");
  { // Read entropy
    Real * table_entropy = subtable["entropy"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
        for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
          size_t idx_flat_input = idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*idx_ni);
          size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*(idx_ni + host_nni(table_idx)*ECENTD));
          Real s_current = table_entropy[idx_flat_input];
          host_table(idx_flat_table) = s_current;
        }
      }
    }
  }

  //printf("Read dpdn.\n");
  { // Read dpdn
    Real * table_dpdn = subtable["dpdn"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
        for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
          size_t idx_flat_input = idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*idx_ni);
          size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*(idx_ni + host_nni(table_idx)*ECDPDN));
          Real dpdn_current = table_dpdn[idx_flat_input];
          host_table(idx_flat_table) = dpdn_current;
        }
      }
    }
  }

  //printf("Read dpdt.\n");
  { // Read dpdt
    Real * table_dpdt = subtable["dpdt"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
        for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
          size_t idx_flat_input = idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*idx_ni);
          size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*(idx_ni + host_nni(table_idx)*ECDPDT));
          Real dpdt_current = table_dpdt[idx_flat_input];
          host_table(idx_flat_table) = dpdt_current;
        }
      }
    }
  }

  //printf("Read dsdn.\n");
  { // Read dsdn
    Real * table_dsdn = subtable["dsdn"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
        for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
          size_t idx_flat_input = idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*idx_ni);
          size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*(idx_ni + host_nni(table_idx)*ECDSDN));
          Real dsdn_current = table_dsdn[idx_flat_input];
          host_table(idx_flat_table) = dsdn_current;
        }
      }
    }
  }

  //printf("Read dsdt.\n");
  { // Read dsdt
    Real * table_dsdt = subtable["dsdt"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_yi=0; idx_yi<host_nyi(table_idx); ++idx_yi) {
        for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
          size_t idx_flat_input = idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*idx_ni);
          size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_yi + host_nyi(table_idx)*(idx_ni + host_nni(table_idx)*ECDSDT));
          Real dsdt_current = table_dsdt[idx_flat_input];
          host_table(idx_flat_table) = dsdt_current;
        }
      }
    }
  }

  // Copy from host to device
  Kokkos::deep_copy(nni, host_nni);
  Kokkos::deep_copy(nyi, host_nyi);
  Kokkos::deep_copy(nt,  host_nt);
  Kokkos::deep_copy(inv_log_ni, host_inv_log_ni);
  Kokkos::deep_copy(inv_yi,     host_inv_yi);
  Kokkos::deep_copy(inv_log_t,  host_inv_log_t);
  Kokkos::deep_copy(Pmin,     host_Pmin);
  Kokkos::deep_copy(offset_ni,    host_offset_ni);
  Kokkos::deep_copy(offset_yi,    host_offset_yi);
  Kokkos::deep_copy(offset_t,     host_offset_t);
  Kokkos::deep_copy(offset_table, host_offset_table);
  Kokkos::deep_copy(ni,     host_ni);
  Kokkos::deep_copy(log_ni, host_log_ni);
  Kokkos::deep_copy(yi, host_yi);
  Kokkos::deep_copy(t,     host_t);
  Kokkos::deep_copy(log_t, host_log_t);

  return success;
}

template<typename LogPolicy>
bool EOSMultiTable<LogPolicy>::Read2DTableFromFile(std::string fname, int table_idx) {
  bool success = true;
  /// Create host mirrors of device arrays to read into, and copy
  HostArray1D<int>::HostMirror  host_nni = create_mirror_view(nni), host_nt = create_mirror_view(nt);
  HostArray1D<Real>::HostMirror host_inv_log_ni = create_mirror_view(inv_log_ni), host_inv_log_t = create_mirror_view(inv_log_t);
  HostArray1D<Real>::HostMirror host_Pmin = create_mirror_view(Pmin);
  HostArray1D<int>::HostMirror  host_offset_ni = create_mirror_view(offset_ni), host_offset_t = create_mirror_view(offset_t), host_offset_table = create_mirror_view(offset_table);
  HostArray1D<Real>::HostMirror host_ni = create_mirror_view(ni), host_log_ni = create_mirror_view(log_ni);
  HostArray1D<Real>::HostMirror host_t = create_mirror_view(t), host_log_t = create_mirror_view(log_t);
  HostArray1D<Real>::HostMirror host_table = create_mirror_view(table);

  // Copy data from device to host
  Kokkos::deep_copy(host_nni, nni);
  Kokkos::deep_copy(host_nt, nt);
  Kokkos::deep_copy(host_inv_log_ni, inv_log_ni);
  Kokkos::deep_copy(host_inv_log_t, inv_log_t);
  Kokkos::deep_copy(host_Pmin, Pmin);
  Kokkos::deep_copy(host_offset_ni, offset_ni);
  Kokkos::deep_copy(host_offset_t, offset_t);
  Kokkos::deep_copy(host_offset_table, offset_table);
  Kokkos::deep_copy(host_ni, ni);
  Kokkos::deep_copy(host_log_ni, log_ni);
  Kokkos::deep_copy(host_t, t);
  Kokkos::deep_copy(host_log_t, log_t);

  // Read subtable
  TableReader::Table subtable;
  auto read_result = subtable.ReadTable(fname);
  if (read_result.error != TableReader::ReadResult::SUCCESS) {
    printf("error opening MultiTable 2D subtable: %s\n",fname.c_str());
    success = false;
    return success;
  }
  // Make sure table has correct dimentions
  if (subtable.GetNDimensions()!=2) {
    printf("MultiTable 2D subtable does not match expected number of dimensions: %ld\n",subtable.GetNDimensions());
    success = false;
    return success;
  }

  // Read scalars
  //printf("Read scalars.\n");
  auto& table_scalars = subtable.GetScalars();
  min_h += table_scalars.at("h_min");

  // Read dims
  //printf("Read dims.\n");
  auto& point_info = subtable.GetPointInfo();
  host_nni(table_idx) = point_info[0].second;
  host_nt(table_idx)  = point_info[1].second;
  
  if (table_idx==0) {
    host_offset_ni(table_idx) = 0;
    host_offset_t(table_idx) = 0;
    host_offset_table(table_idx) = 0;
  }

  if (table_idx!=(n_tables_3D+n_tables_2D-1)){
    host_offset_ni(table_idx+1) = host_offset_ni(table_idx) + host_nni(table_idx);
    host_offset_t(table_idx+1) = host_offset_t(table_idx) + host_nt(table_idx);
    host_offset_table(table_idx+1) = host_offset_table(table_idx) + host_nni(table_idx)*host_nt(table_idx)*ECNVARS;
  }

  //printf("Read nb.\n");
  { // read nb
    Real * table_ni = subtable["ni"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      host_ni(host_offset_ni(table_idx)+idx_ni)     = table_ni[idx_ni];
      host_log_ni(host_offset_ni(table_idx)+idx_ni) = log2_(host_ni(host_offset_ni(table_idx)+idx_ni));
    }
    host_inv_log_ni(table_idx) = 1.0/(host_log_ni(host_offset_ni(table_idx)+1) - host_log_ni(host_offset_ni(table_idx)+0));
  }

  //printf("Read T.\n");
  { // read T
    Real * table_t = subtable["t"];
    for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
      host_t(host_offset_t(table_idx)+idx_t)     = table_t[idx_t];
      host_log_t(host_offset_t(table_idx)+idx_t) = log2_(host_t(host_offset_t(table_idx)+idx_t));
    }
    
    host_inv_log_t(table_idx) = 1.0/(host_log_t(host_offset_t(table_idx)+1) - host_log_t(host_offset_t(table_idx)+0));
  }

  //printf("Read pressure.\n");
  { // Read pressure
    Real * table_press = subtable["pressure"];
    Real Pmin_read = table_press[0];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
        size_t idx_flat_input = idx_t + host_nt(table_idx)*idx_ni;
        Real p_current = table_press[idx_flat_input];
        if (p_current < Pmin_read) {Pmin_read = p_current;}
      }
    }

    if (Pmin_read<0.0) {
      Pmin(table_idx) = -Pmin_read * (1.0 + Pmin_fac);
    } else {
      Pmin(table_idx) = 0.0;
    }

    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
        size_t idx_flat_input = idx_t + host_nt(table_idx)*idx_ni;
        size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_ni + host_nni(table_idx)*ECLOGP);
        Real p_current = table_press[idx_flat_input];
        host_table(idx_flat_table) = log2_(p_current + Pmin(table_idx));
      }
    }
  }

  //printf("Read energy.\n");
  { // Read energy
    Real * table_energy = subtable["energy"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
        size_t idx_flat_input = idx_t + host_nt(table_idx)*idx_ni;
        size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_ni + host_nni(table_idx)*ECLOGE);
        Real e_current = table_energy[idx_flat_input];
        host_table(idx_flat_table) = log2_(e_current);
      }
    }
  }

  //printf("Read entropy.\n");
  { // Read entropy
    Real * table_entropy = subtable["entropy"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
        size_t idx_flat_input = idx_t + host_nt(table_idx)*idx_ni;
        size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_ni + host_nni(table_idx)*ECENTD);
        Real s_current = table_entropy[idx_flat_input];
        host_table(idx_flat_table) = s_current;
      }
    }
  }

  //printf("Read dpdn.\n");
  { // Read dpdn
    Real * table_dpdn = subtable["dpdn"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
        size_t idx_flat_input = idx_t + host_nt(table_idx)*idx_ni;
        size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_ni + host_nni(table_idx)*ECDPDN);
        Real dpdn_current = table_dpdn[idx_flat_input];
        host_table(idx_flat_table) = dpdn_current;
      }
    }
  }

  //printf("Read dpdt.\n");
  { // Read dpdt
    Real * table_dpdt = subtable["dpdt"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
        size_t idx_flat_input = idx_t + host_nt(table_idx)*idx_ni;
        size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_ni + host_nni(table_idx)*ECDPDT);
        Real dpdt_current = table_dpdt[idx_flat_input];
        host_table(idx_flat_table) = dpdt_current;
      }
    }
  }

  //printf("Read dsdn.\n");
  { // Read dsdn
    Real * table_dsdn = subtable["dsdn"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
        size_t idx_flat_input = idx_t + host_nt(table_idx)*idx_ni;
        size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_ni + host_nni(table_idx)*ECDSDN);
        Real dsdn_current = table_dsdn[idx_flat_input];
        host_table(idx_flat_table) = dsdn_current;
      }
    }
  }

  //printf("Read dsdt.\n");
  { // Read dsdt
    Real * table_dsdt = subtable["dsdt"];
    for (size_t idx_ni=0; idx_ni<host_nni(table_idx); ++idx_ni) {
      for (size_t idx_t=0; idx_t<host_nt(table_idx); ++idx_t) {
        size_t idx_flat_input = idx_t + host_nt(table_idx)*idx_ni;
        size_t idx_flat_table = host_offset_table(table_idx) + idx_t + host_nt(table_idx)*(idx_ni + host_nni(table_idx)*ECDSDT);
        Real dsdt_current = table_dsdt[idx_flat_input];
        host_table(idx_flat_table) = dsdt_current;
      }
    }
  }

  // Copy from host to device
  Kokkos::deep_copy(nni, host_nni);
  Kokkos::deep_copy(nt,  host_nt);
  Kokkos::deep_copy(inv_log_ni, host_inv_log_ni);
  Kokkos::deep_copy(inv_log_t,  host_inv_log_t);
  Kokkos::deep_copy(Pmin,     host_Pmin);
  Kokkos::deep_copy(offset_ni,    host_offset_ni);
  Kokkos::deep_copy(offset_t,     host_offset_t);
  Kokkos::deep_copy(offset_table, host_offset_table);
  Kokkos::deep_copy(ni,     host_ni);
  Kokkos::deep_copy(log_ni, host_log_ni);
  Kokkos::deep_copy(t,     host_t);
  Kokkos::deep_copy(log_t, host_log_t);
  return success;
}

template<typename LogPolicy>
bool EOSMultiTable<LogPolicy>::ReadTUnionTableFromFile(std::string fname) {
  bool success = true;
  HostArray1D<Real>::HostMirror host_t_union = create_mirror_view(t_union), host_log_t_union = create_mirror_view(log_t_union);

  Kokkos::deep_copy(host_t_union, t_union);
  Kokkos::deep_copy(host_log_t_union, log_t_union);

  // Read subtable
  TableReader::Table subtable;
  auto read_result = subtable.ReadTable(fname);
  if (read_result.error != TableReader::ReadResult::SUCCESS) {
    printf("error opening T union subtable: %s\n",fname.c_str());
    success = false;
    return success;
  }
  // Make sure table has correct dimentions
  if (subtable.GetNDimensions()!=1) {
    printf("T union subtable does not match expected number of dimensions: %ld\n",subtable.GetNDimensions());
    success = false;
    return success;
  }
  
  //printf("Read T.\n");
  { // read T
    Real * table_t = subtable["t"];
    for (size_t idx_t=0; idx_t<n_T_union; ++idx_t) {
      host_t_union(idx_t)     = table_t[idx_t];
      host_log_t_union(idx_t) = log2_(host_t_union(idx_t));
    }
  }

  Kokkos::deep_copy(t_union,     host_t_union);
  Kokkos::deep_copy(log_t_union, host_log_t_union);

  return success;
}

template class EOSMultiTable<NormalLogs>;
template class EOSMultiTable<NQTLogs>;

} // namespace primitive