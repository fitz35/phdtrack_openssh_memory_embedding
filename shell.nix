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

    # imbalancing
    buildPythonPackage rec {
      pname = "imbalanced-learn";
      version = "0.11.0";
      src = fetchPypi {
        inherit pname version;
        sha256 = "sha256-7582ae8858e6db0b92fef97dd08660a18297ee128d78c2abdc006b8bd86b8fdc";
      };
      doCheck = false;
      propagatedBuildInputs = [
        # Specify dependencies
        pkgs.python311Packages.numpy
        pkgs.python311Packages.scipy
        pkgs.python311Packages.scikit-learn
      ];
    }

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