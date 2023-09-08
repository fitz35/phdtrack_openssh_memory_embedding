{ pkgs ? import <nixpkgs> {} }:

let
  my-python-packages = ps: with ps; [
    # python packages
    python-dotenv
    psutil # for system monitoring
    pandas 
    numpy

    seaborn # for plotting
    matplotlib # for plotting
    scikit-learn # for machine learning

    # quality checks
    mypy
    pandas-stubs
    types-psutil
  ];
  my-python = pkgs.python311.withPackages my-python-packages;
in
pkgs.mkShell {
  packages = [
    # packages
    my-python

    
  ];
}