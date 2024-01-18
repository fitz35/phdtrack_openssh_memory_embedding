if __name__ == "__main__":
    outputs_sizes = [8, 16, 100]
    window_character_sizes = [8, 16]
    word_character_sizes = [2, 4] # size of the word in bytes


    print("\\begin{table}[!ht]")
    print("\t\\centering")
    print("\t\\begin{tabular}{|c|c|c|}")
    print("\t\\hline")
    print("\t\\textbf{Word size} & \\textbf{window character size} & \\textbf{Embedding dimension} \\\\ \\hline")

    for output_size in outputs_sizes:
        for window_character_size in window_character_sizes:
            for word_character_size in word_character_sizes:
                    print(f"\t\t{word_character_size} & {window_character_size} & {output_size} \\\\ \\hline")

    print("\t\\end{tabular}")
    print("\t\\caption{Hyperparameters for the Transformers model.}")
    print("\t\\label{tab:transformers_hyperparams}")
    print("\\end{table}")
                    