"""
Transformer Encoder "From Scratch"
Disciplina: Tópicos em Inteligência Artificial – 2026.1
"""

import numpy as np
import pandas as pd

# Configurações do modelo
D_MODEL = 64
BATCH_SIZE = 1

# Passo 1: Preparação dos dados

vocabulario = {
    "o": 0,
    "victor": 1,
    "tirou": 2,
    "dez": 3,
    "na": 4,
    "prova": 5,
    "do": 6,
    "dimmy": 7,
}

df_vocabulario = pd.DataFrame(
    list(vocabulario.items()), columns=["palavra", "id"]
)
print("Vocabulário:")
print(df_vocabulario)
print()

frase_entrada = ["o", "victor", "tirou", "dez", "na", "prova", "do", "dimmy"]
sequencia_ids = [vocabulario[palavra] for palavra in frase_entrada]
print(f"Frase: {frase_entrada}")
print(f"IDs:   {sequencia_ids}\n")

vocab_size = len(vocabulario)
sequence_length = len(sequencia_ids)

np.random.seed(42)
tabela_embeddings = np.random.randn(vocab_size, D_MODEL)

# Monta o tensor de entrada X com shape (BatchSize, SequenceLength, D_MODEL)
embeddings_sequencia = tabela_embeddings[sequencia_ids]
X = embeddings_sequencia[np.newaxis, :, :]

print(f"Shape do tensor de entrada X: {X.shape}")
print(f"Esperado: ({BATCH_SIZE}, {sequence_length}, {D_MODEL})")