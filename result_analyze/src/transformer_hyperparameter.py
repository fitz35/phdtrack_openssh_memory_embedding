
if __name__ == "__main__":
    word_character_sizes = [16, 8] # size of the word in bytes (take care to not overflow f64, so max 8 bytes, ie 16 characters)
    embedding_dims = [8, 16] # output of the embedding (result size)
    transformer_units = [2, 4] # dimension of the transformer units (see .md)
    num_heads = [2, 4] # attention heads
    num_transformer_layers = [2, 4] # number of transformer layers
    dropout_rates = [0.1, 0.3]
    activations = ["relu"]

    zipped = zip(zip(zip(transformer_units, num_heads), num_transformer_layers), dropout_rates)

    print("\\begin{table}[!ht]")
    print("\t\\centering")
    print("\t\\begin{tabular}{|p{2cm}|p{2cm}|p{2cm}|p{2cm}|p{2cm}|p{2cm}|}")
    print("\t\\hline")
    print("\t \\textbf{Word size} & \\textbf{Embedding dimension} & \\textbf{Transformer units} & \\textbf{Num heads} & \\textbf{Num layers} &  \\textbf{Dropout rate}  \\\\ \\hline")

    for (((transformer_unit, num_head), num_transformer_layer), dropout_rate) in zipped:
        for word_character_size in word_character_sizes:
            for embedding_dim in embedding_dims:
            
                for activation in activations:
                    print(f"\t\t {word_character_size} & {embedding_dim} & {transformer_unit} & {num_head} &  {num_transformer_layer} &  {dropout_rate} \\\\ \\hline")

    print("\t\\end{tabular}")
    print("\t\\caption{Hyperparameters for the Transformers model.}")
    print("\t\\label{tab:transformers_hyperparams}")
    print("\\end{table}")
                    