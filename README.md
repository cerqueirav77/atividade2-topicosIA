# Transformer Encoder From Scratch — LAB P2

## Descrição

Este repositório contém a implementação do Forward Pass completo de um bloco Encoder
da arquitetura Transformer, baseada no paper *"Attention Is All You Need"* (Vaswani et al., 2017).

O sistema recebe uma frase como entrada, converte as palavras em vetores numéricos
(embeddings) e processa a sequência através de 6 camadas idênticas do Encoder,
produzindo ao final a representação densa contextualizada **Z** — onde cada token
carrega informação sobre todos os outros tokens da frase.

A implementação utiliza exclusivamente `numpy` e `pandas`, sem nenhuma biblioteca
de deep learning, demonstrando como as operações matemáticas do Transformer
funcionam por baixo dos panos.

## Como Rodar

**Pré-requisitos:** Python 3.x

1. Crie e ative um ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale as dependências:
```bash
pip install numpy pandas
```

3. Execute o encoder:
```bash
python3 encoder.py
```

4. Desative o ambiente virtual ao terminar (opcional):
```bash
deactivate
```

## Estrutura do Encoder

Cada uma das 6 camadas executa o seguinte fluxo:
```
X_att   = SelfAttention(X)
X_norm1 = LayerNorm(X + X_att)
X_ffn   = FFN(X_norm1)
X_out   = LayerNorm(X_norm1 + X_ffn)
X       = X_out  →  vira input da próxima camada
```

## Explicação da Normalização (√dₖ)

Durante o cálculo da atenção, o produto escalar `QKᵀ` cresce em magnitude
proporcional à dimensão `dₖ`. Para valores grandes de `dₖ`, isso empurra o
softmax para regiões de saturação, onde os gradientes ficam próximos de zero
— dificultando o treinamento.

Dividir por `√dₖ` mantém a variância dos scores próxima de 1, independente
da dimensão do modelo, garantindo gradientes mais úteis e treinamento mais estável.

## Exemplo de Input / Output

**Frase de entrada:**
```
["o", "victor", "tirou", "dez", "na", "prova", "do", "dimmy"]
```

**Vocabulário mapeado:**
```
{"o": 0, "victor": 1, "tirou": 2, "dez": 3, "na": 4, "prova": 5, "do": 6, "dimmy": 7}
```

**Shapes ao longo do processamento:**

| Etapa                        | Shape         |
|------------------------------|---------------|
| Tensor de entrada X          | (1, 8, 64)    |
| Saída da Self-Attention       | (1, 8, 64)    |
| Saída do Add & LayerNorm 1   | (1, 8, 64)    |
| Saída da FFN                 | (1, 8, 64)    |
| Saída do Add & LayerNorm 2   | (1, 8, 64)    |
| Representação final Z        | (1, 8, 64)    |

A dimensão é preservada em todas as camadas: **(Batch=1, Tokens=8, D_MODEL=64)**.

## Validação de Sanidade
```
Camada 1 — shape: (1, 8, 64)
Camada 2 — shape: (1, 8, 64)
Camada 3 — shape: (1, 8, 64)
Camada 4 — shape: (1, 8, 64)
Camada 5 — shape: (1, 8, 64)
Camada 6 — shape: (1, 8, 64)

Representação final Z — shape: (1, 8, 64)
Validação: entrada e saída com mesmas dimensões? True
```

## Fórmulas de Referência

**Scaled Dot-Product Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Conexão Residual + LayerNorm:**

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Feed-Forward Network:**

$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)W_2 + b_2$$

## Referência

Vaswani, A. et al. **Attention Is All You Need**, 2017.
https://arxiv.org/abs/1706.03762

## Auxiliado por

Implementação desenvolvida com auxílio do Claude (Anthropic) como ferramenta
de suporte ao aprendizado, conforme permitido pelo contrato pedagógico da disciplina.