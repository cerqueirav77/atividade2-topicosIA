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

# Passo 2.1: Scaled Dot-Product Attention 

np.random.seed(7)
W_query = np.random.randn(D_MODEL, D_MODEL)
W_key   = np.random.randn(D_MODEL, D_MODEL)
W_value = np.random.randn(D_MODEL, D_MODEL)


def softmax(matriz: np.ndarray) -> np.ndarray:
    """Aplica softmax linha a linha com estabilidade numérica."""
    valores_deslocados = matriz - np.max(matriz, axis=-1, keepdims=True)
    exponenciais = np.exp(valores_deslocados)
    return exponenciais / np.sum(exponenciais, axis=-1, keepdims=True)


def self_attention(X: np.ndarray) -> np.ndarray:
    """
    Calcula o Scaled Dot-Product Attention.

    Args:
        X (ndarray): Tensor de entrada com shape (Batch, Seq, D_MODEL).

    Returns:
        ndarray: Tensor de saída com shape (Batch, Seq, D_MODEL).
    """
    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value

    dimensao_k = K.shape[-1]
    fator_escala = np.sqrt(dimensao_k)

    scores = Q @ K.transpose(0, 2, 1)
    scores_normalizados = scores / fator_escala
    pesos_atencao = softmax(scores_normalizados)
    saida_atencao = pesos_atencao @ V

    return saida_atencao


X_att = self_attention(X)
print(f"\nSaída da Self-Attention — shape: {X_att.shape}")

# Passo 2.2: Conexão Residual e Layer Normalization

EPSILON = 1e-6


def layer_norm(tensor: np.ndarray) -> np.ndarray:
    """
    Aplica Layer Normalization no último eixo (features).

    Args:
        tensor (ndarray): Tensor de entrada com shape (Batch, Seq, D_MODEL).

    Returns:
        ndarray: Tensor normalizado com mesma shape.
    """
    media = np.mean(tensor, axis=-1, keepdims=True)
    variancia = np.var(tensor, axis=-1, keepdims=True)
    tensor_normalizado = (tensor - media) / np.sqrt(variancia + EPSILON)
    return tensor_normalizado


def adicionar_e_normalizar(tensor_entrada: np.ndarray, saida_sublayer: np.ndarray) -> np.ndarray:
    """Aplica a conexão residual e normaliza o resultado."""
    tensor_residual = tensor_entrada + saida_sublayer
    return layer_norm(tensor_residual)


X_norm1 = adicionar_e_normalizar(X, X_att)
print(f"Saída do Add & LayerNorm 1 — shape: {X_norm1.shape}")