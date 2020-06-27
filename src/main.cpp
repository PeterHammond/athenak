//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//////////////////////////////////////// AthenaXXX Main Program ////////////////////////////////////
//! \file main.cpp
//  \brief AthenaXXX main program
//
// Based on the Athena (Cambridge version) and Athena++ MHD codes.  Athena was  originally written
// in 2002-2005 by Jim Stone, Tom Gardiner, and Peter Teuben, with many important contributions by
// many other developers after that, i.e. 2005-2014.
//
// Athena++ was started in Jan 2014.  The core design was finished during 4-7/2014 at the
// KITP by Jim Stone.  GR was implemented by Chris White and AMR by Kengo Tomida during
// 2014-2016.  Contributions from many others have continued to the present.
//
// AthenaXXX is an outgrowth of the Athena-Parthenon collaboration, and is a completely new
// implementation based on the Kokkos performance-portability library (now an external module)
//==================================================================================================

#include <cstdlib>
#include <iostream>   // cout, endl
#include <string>     // string

#include "athena.hpp"
#include "utils/utils.hpp"

//--------------------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief AthenaK main program

int main(int argc, char *argv[]) {
  std::string athena_version = "version 20.0 - August 2020";
  char *input_filename = nullptr, *restart_filename = nullptr, *prundir = nullptr;
  int res_flag  = 0;  // set to 1 if -r        argument is on cmdline
  int narg_flag = 0;  // set to 1 if -n        argument is on cmdline
  int iarg_flag = 0;  // set to 1 if -i <file> argument is on cmdline
  int mesh_flag = 0;  // set to <nproc> if -m <nproc> argument is on cmdline
  int wtlim = 0;
  std::uint64_t mbcnt = 0;

  //--- Step 1. ------------------------------------------------------------------------------------
  // Initialize environment

#if MPI_PARALLEL_ENABLED
#if OPENMP_PARALLEL_ENABLED
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for the hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv
              << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
#endif  // OPENMP_PARALLEL_ENABLED
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(global_variable::my_rank))) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    MPI_Finalize();
    return(0);
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_size failed." << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no MPI
  global_variable::my_rank = 0;
  global_variable::nranks  = 1;
#endif  // MPI_PARALLEL_ENABLED

  //--- Step 2. ------------------------------------------------------------------------------------
  // Check for command line options and respond.

  for (int i=1; i<argc; i++) {
    // If argv[i] is a 2 character string of the form "-?" then:
    if (*argv[i] == '-'  && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0') {
      // check validity of command line options + arguments:
      char opt_letter = *(argv[i]+1);
      switch(opt_letter) {
        // options that do not take arguments:
        case 'n':
        case 'c':
        case 'h':
          break;
          // options that require arguments:
        default:
          if ((i+1 >= argc) // flag is at the end of the command line options
              || (*argv[i+1] == '-') ) { // flag is followed by another flag
            if (global_variable::my_rank == 0) {
              std::cout << "### FATAL ERROR in main" << std::endl
                        << "-" << opt_letter << " must be followed by a valid argument\n";
#if MPI_PARALLEL_ENABLED
              MPI_Finalize();
#endif
              return(0);
            }
          }
      }
      switch(*(argv[i]+1)) {
        case 'i':                      // -i <input_filename>
          input_filename = argv[++i];
          iarg_flag = 1;
          break;
        case 'r':                      // -r <restart_file>
          res_flag = 1;
          restart_filename = argv[++i];
          break;
        case 'd':                      // -d <run_directory>
          prundir = argv[++i];
          break;
        case 'n':
          narg_flag = 1;
          break;
        case 'm':                      // -m <nproc>
          mesh_flag = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
          break;
        case 't':                      // -t <hh:mm:ss>
          int wth, wtm, wts;
          std::sscanf(argv[++i], "%d:%d:%d", &wth, &wtm, &wts);
          wtlim = wth*3600 + wtm*60 + wts;
          break;
        case 'c':
          if (global_variable::my_rank == 0) ShowConfig();
#if MPI_PARALLEL_ENABLED
          MPI_Finalize();
#endif
          return(0);
          break;
        case 'h':
        default:
          if (global_variable::my_rank == 0) {
            std::cout << "Athena++ " << athena_version << std::endl;
            std::cout << "Usage: " << argv[0] << " [options] [block/par=value ...]\n";
            std::cout << "Options:" << std::endl;
            std::cout << "  -i <file>       specify input file [athinput]\n";
            std::cout << "  -r <file>       restart with this file\n";
            std::cout << "  -d <directory>  specify run dir [current dir]\n";
            std::cout << "  -n              parse input file and quit\n";
            std::cout << "  -c              show configuration and quit\n";
            std::cout << "  -m <nproc>      output mesh structure and quit\n";
            std::cout << "  -t hh:mm:ss     wall time limit for final output\n";
            std::cout << "  -h              this help\n";
            ShowConfig();
          }
#if MPI_PARALLEL_ENABLED
          MPI_Finalize();
#endif
          return(0);
          break;
      }
    } // else if argv[i] not of form "-?" ignore it here (tested in ModifyFromCmdline)
  }

  if (restart_filename == nullptr && input_filename == nullptr) {
    // no input file is given
    std::cout << "### FATAL ERROR in main" << std::endl
              << "No input file or restart file is specified." << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Step 3. ------------------------------------------------------------------------------------
  // Construct object to store input parameters, then parse input file and command line.
  // With MPI, the input is read by every process in parallel using MPI-IO.


  //--- Step 4. ------------------------------------------------------------------------------------
  // Construct and initialize Mesh


  //--- Step 5. ------------------------------------------------------------------------------------
  // Construct and initialize TaskList

  //--- Step 6. ------------------------------------------------------------------------------------
  // Set initial conditions by calling problem generator, or reading restart file

  //--- Step 7. ------------------------------------------------------------------------------------
  // Change to run directory, initialize outputs object, and make output of ICs

  //--- Step 8. ------------------------------------------------------------------------------------

  //--- Step 9. ------------------------------------------------------------------------------------
  // Make the final outputs

  //--- Step 10. -----------------------------------------------------------------------------------
  // Terminate

#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return(0);
}
