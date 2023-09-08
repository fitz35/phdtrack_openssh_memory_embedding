# phdtrack_openssh_memory_embedding

## python interpreter

To configure vscode python interpreter, open the nix-shell and run `which python` to have the path, and paste it inside the interpreter settings.

## Installation

Use nix [shell](https://ryantm.github.io/nixpkgs/languages-frameworks/python/).

### Work with no packaged python module

[Here](https://github.com/NixOS/nixpkgs/blob/49829a9adedc4d2c1581cc9a4294ecdbff32d993/doc/languages-frameworks/python.section.md#how-to-consume-python-modules-using-pip-in-a-virtual-environment-like-i-am-used-to-on-other-operating-systems-how-to-consume-python-modules-using-pip-in-a-virtual-environment-like-i-am-used-to-on-other-operating-systems) is the documentation. The trick is to launch a .venv environnement in the [shell](shell.nix) hook wich call pip and the [requirements](requirements.txt) file.

## Mypy

mypy must be run in the `src` folder.
