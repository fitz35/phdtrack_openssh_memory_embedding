with import <nixpkgs> {};
let
  myPythonEnv = python311.withPackages (ps: [
    
    
  ]);
in mkShell {
  packages = [
    myPythonEnv

    mypy
  ];
}