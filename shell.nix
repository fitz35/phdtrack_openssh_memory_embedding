{ pkgs ? import <nixpkgs> {} }:

let
  my-python-packages = ps: with ps; [
    # python packages
    python-dotenv
    psutil
    pandas
    numpy

    mypy
  ];
  my-python = pkgs.python311.withPackages my-python-packages;
in
pkgs.mkShell {
  packages = [
    # packages
    my-python

    
  ];
}