
if __name__ == "__main__":
    word_character_sizes = [16, 8] # size of the word in bytes (take care to not overflow f64, so max 8 bytes, ie 16 characters)
    embedding_dims = [8, 16] # output of the embedding (result size)
    transformer_units = [2, 4] # dimension of the transformer units (see .md)
    num_heads = [2, 4] # attention heads
    num_transformer_layers = [2, 4] # number of transformer layers
    dropout_rates = [0.1, 0.3]
    activations = ["relu"]

    zipped = zip(zip(zip(transformer_units, num_heads), num_transformer_layers), dropout_rates)

    header_lines = "\t\\begin{tabular}{|c"
    word_character_size_line = "\t \\textbf{Word size} "
    embedding_dim_line = "\t \\textbf{Embedding dimension} "
    transformer_unit_line = "\t \\textbf{Transformer units} "
    num_head_line = "\t \\textbf{Num heads} "
    num_transformer_layer_line = "\t \\textbf{Num layers} "
    dropout_rate_line = "\t \\textbf{Dropout rate} "

    print("\\begin{table}[!ht]")
    print("\t\\centering")

    for (((transformer_unit, num_head), num_transformer_layer), dropout_rate) in zipped:
        for word_character_size in word_character_sizes:
            for embedding_dim in embedding_dims:
            
                for activation in activations:
                    header_lines += "|c"
                    word_character_size_line += f"& {word_character_size} "
                    embedding_dim_line += f"& {embedding_dim} "
                    transformer_unit_line += f"& {transformer_unit} "
                    num_head_line += f"& {num_head} "
                    num_transformer_layer_line += f"& {num_transformer_layer} "
                    dropout_rate_line += f"& {dropout_rate} "
     
    word_character_size_line += "\\\\ \\hline"
    embedding_dim_line += "\\\\ \\hline"
    transformer_unit_line += "\\\\ \\hline"
    num_head_line += "\\\\ \\hline"
    num_transformer_layer_line += "\\\\ \\hline"
    dropout_rate_line += "\\\\ \\hline"
    header_lines += "|}\\hline"

    print(header_lines)
    print(word_character_size_line)
    print(embedding_dim_line)
    print(transformer_unit_line)
    print(num_head_line)
    print(num_transformer_layer_line)
    print(dropout_rate_line)

    print("\t\\end{tabular}")
    print("\t\\caption{Hyperparameters for the Transformers model.}")
    print("\t\\label{tab:transformers_hyperparams}")
    print("\\end{table}")
                    