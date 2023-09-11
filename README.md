# phdtrack_openssh_memory_embedding

## python interpreter

To configure vscode python interpreter, open the nix-shell and run `which python` to have the path, and paste it inside the interpreter settings.

## Installation

Use nix [shell](https://ryantm.github.io/nixpkgs/languages-frameworks/python/).

### Work with no packaged python module

[Here](https://github.com/NixOS/nixpkgs/blob/49829a9adedc4d2c1581cc9a4294ecdbff32d993/doc/languages-frameworks/python.section.md#how-to-consume-python-modules-using-pip-in-a-virtual-environment-like-i-am-used-to-on-other-operating-systems-how-to-consume-python-modules-using-pip-in-a-virtual-environment-like-i-am-used-to-on-other-operating-systems) is the documentation. The trick is to launch a .venv environnement in the [shell](shell.nix) hook wich call pip and the [requirements](requirements.txt) file.

## Mypy

mypy must be run in the `src` folder.

# Commands

Generate data (mem_to_graph) :

```shell
cargo run -- -p semantic-embedding-dtn -d /root/phdtrack/phdtrack_data/Training -d /root/phdtrack/phdtrack_data/Performance_Test -d /root/phdtrack/phdtrack_data/Validation -o /root/phdtrack/phdtrack_project_3/src/mem_to_graph/data/semantic_dts_embedding &

cargo run -- -p statistic-embedding-dtn -d /root/phdtrack/phdtrack_data/Training -d /root/phdtrack/phdtrack_data/Performance_Test -d /root/phdtrack/phdtrack_data/Validation -o /root/phdtrack/phdtrack_project_3/src/mem_to_graph/data/statistic_dts_embedding &
```

Run python quality check :

```shell
nix-shell
cd src
python3 embedding_quality_main.py -otr training -ots validation -d /root/phdtrack/phdtrack_project_3/src/mem_to_graph/data/statistic_dts_embedding

python3 embedding_quality_main.py -otr training -ots validation -d /root/phdtrack/phdtrack_project_3/src/mem_to_graph/data/semantic_dts_embedding
```
