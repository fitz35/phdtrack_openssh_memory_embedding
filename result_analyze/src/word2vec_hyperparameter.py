if __name__ == "__main__":
    outputs_sizes = [8, 16, 100]
    window_character_sizes = [8, 16]
    word_character_sizes = [2, 4] # size of the word in bytes

    header_line = "\t\\begin{tabular}{|c"
    outputs_size_line = "\t \\textbf{Embedding dimension} "
    window_character_size_line = "\t \\textbf{Window character size} "
    word_character_size_line = "\t \\textbf{Word size} "

    print("\\begin{table}[!ht]")
    print("\t\\centering")

    for output_size in outputs_sizes:
        for window_character_size in window_character_sizes:
            for word_character_size in word_character_sizes:
                outputs_size_line += f"& {output_size} "
                window_character_size_line += f"& {window_character_size} "
                word_character_size_line += f"& {word_character_size} "
                header_line += "|c"

    outputs_size_line += "\\\\ \\hline"
    window_character_size_line += "\\\\ \\hline"
    word_character_size_line += "\\\\ \\hline"
    header_line += "|}\\hline"

    print(header_line)
    print(outputs_size_line)
    print(window_character_size_line)
    print(word_character_size_line)

    print("\t\\end{tabular}")
    print("\t\\caption{Hyperparameters for the Word2Vec model.}")
    print("\t\\label{tab:Word2Vec_hyperparams}")
    print("\\end{table}")
                    