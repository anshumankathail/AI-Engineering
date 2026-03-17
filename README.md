# Transformer Architecture

<p align="center">
  <img src="assets/transformer.png" width="400"/>
</p>

## 1. Overview

The Transformer is a sequence-to-sequence model that processes input data in parallel using attention mechanisms instead of recurrence

A Transformer models:

```
P(y_t | y_{<t}, x)
```

## 2. Embedding Phase

### Tokenization

Input sentence tokens:

```
[I, love, AI]
```

### Input Embedding

> Each token is mapped to a vector using a learned embedding matrix:

```
E ∈ ℝ^(vocab_size × d_model)
```

Input:
```
X ∈ ℝ^{n × d_model}
```
- n = number of tokens

- d_model = length of embedding vector

### Positional Encoding

> Adds order information to tokens

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Final input:

```
X = X_embed + X_pos
```

## 3. Encoder

> Nx encoder layers are stacked where each encoder is fed output from the previous encoder as input

> Encoder processes the full input sequence in parallel to build contextual representations

### 3.1 Self-Attention

> Captures relationships between input tokens

> Each token looks at all other tokens and decides what is relevant

```
Q = X W_Q
K = X W_K
V = X W_V
```

Shapes:

```
X ∈ ℝ^{n × d_model}
W_Q, W_K, W_V ∈ ℝ^{d_model × d_k}
```

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- Output becomes context-aware

### 3.2 Multi-Head Attention

> Learns multiple relationships simultaneously

> Each head captures a different relationship (semantics, grammar, syntax, etc.)

```
head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
```

```
MultiHead = Concat(head_1, ..., head_h) W_O
```

```
W_O ∈ ℝ^{h·d_k × d_model}
```
- d_model = h * d_k

### 3.3 Add & Norm

```
X = LayerNorm(X + MultiHead(X))
```

- Adding to input fed at the previous layer prevents vanishing gradients

- LayerNorm stabilizes training

- LayerNorm has 2 learnable parameters (γ, β)

### 3.4 Feed Forward Network (FFN)
> Processes each token independently to refine features

> Projects to higher dimensional space (to learn representations of each token) then back to original

```
FFN(X) = σ(X W1 + b1) W2 + b2
```

```
W_1 ∈ ℝ^{d_model × d_ff}
W_2 ∈ ℝ^{d_ff × d_model}
```

- Usually, d_ff = 4 * d_model

- Applied independently to each token

- Adds non-linearity

### 3.5 Add & Norm

```
Enc_out = LayerNorm(X + FFN(X))
```

## 4. Decoder

> Nx decoder layers are stacked where each decoder is fed output from the previous decoder as input

> During training phase, decoder is fed the target sentence (shifted right i.e. `<end>` token is removed) and next token prediction for each token runs in parallel

> During inference phase, decoder generates next token one step at a time using previously generated tokens and encoder context

Decoder has two attention mechanisms:
- Masked Self-Attention (output → output)

- Cross Attention (output → input)

### 4.1 Masked Self-Attention

> Prevents access to future tokens

```
Q = X_dec W_Q
K = X_dec W_K
V = X_dec W_V
```

```
Mask (M): [0  -∞ -∞]
          [0   0 -∞]
          [0   0  0]
```

```
Attention = softmax((QK^T + M) / √d_k) V
```

### 4.2 Add & Norm

```
X = LayerNorm(X + FFN(X))
```

### 4.3 Cross Attention

> Helps learn relationship between input and output

```
Q = X_dec W_Q
K = Enc_out W_K
V = Enc_out W_V
```

```
Attention = softmax(QK^T / √d_k) V
```

### 4.4 Add & Norm

```
X = LayerNorm(X + FFN(X))
```

### 4.5 Feed Forward Network

```
FFN(X) = σ(X W_1 + b_1) W_2 + b_2
```

### 4.6 Add & Norm

```
Z = LayerNorm(X + FFN(X))
```

## 5. Output Layer

```
Logits = Z W + b
```

```
Z ∈ ℝ^{n × d_model}

W ∈ ℝ^{d_model × vocab_size}
```

```
P = softmax(Logits)
```

- Converts to probability distribution over vocabulary

## 6. Training Phase

> Next token prediction in training phase is not autoregressive — everything runs in parallel

> It is because we already know the next token and we only need to compute the loss between predicted and actual output

Objective (Minimize):

```
L = -∑ log P(y_t | y_<t, x)
```

- Model learns next-token prediction

## 7. Inference Phase

> This phase is autoregressive

Steps:
- Start with `<start>`

- Predict next token

- Append token

- Repeat

- Stop at `<end>`

## 8. KV Caching

> For masked self attention, compute Q for the new token and append only new K, V (K, V for previous tokens remain unchanged)

> For cross-attention, K and V are computed once and reused for all steps

At step t:

```
Q ∈ ℝ^{1 × d_k}
K, V ∈ ℝ^{t-1 × d_k}
```

Append:

```
K = [K; k_t]
V = [V; v_t]
```

- Reduces complexity from O(n²) → O(n)

## 9. Beam Search

> Chooses top_k next token predictions at each step

> At the end, selects best sentence based on final score

Score:

```
Score = sum(log probabilities) / length^α
```

- length^α is used to reduce the impact of sentence length on the final Score

- This is because, longer sentences can have higher value of sum(log probabilities) 

## 10. Key Concepts Summary

- Encoder: Understands input

- Decoder: Generates output

- Attention: Focus on relevant info

- Self-Attention: token-to-token relationship within the same sequence

- Cross-Attention: token-to-token relationship b/w input & output

- FFN: per-token transformation

- Masking: prevents future leakage

---

<br>
<br>

# Other Concepts

## Special Tokens

> Special tokens are predefined tokens added to the input sequence to guide the transformer’s behavior for specific tasks

> Special tokens keeps the input consistent and helps the transformer understand the input structure


### CLS (Classification Token) 
> Placed at the beginning of the input sequence

- Represents the entire sequence  
- Final hidden state of this token is used for classification tasks  

Example: <br>
[CLS] I love AI


Used in:
- Text classification  
- Sentiment analysis  

---

### SEP (Separator Token) 
> Used to separate different segments of text  

Example: <br>
[CLS] Sentence A [SEP] Sentence B [SEP]


Used in:
- Sentence pair classification  
- Question answering  

---

### PAD (Padding Token)
> Used to make all sequences in a batch the same length  

- Ensures efficient parallel processing  
- Ignored during attention using masking  

Example: <br>
[I, love, AI, PAD, PAD]


---

### Truncation
> Cuts longer sequences to a fixed maximum length  

- Prevents memory overflow  
- Ensures consistent input size  

---

### MASK Token
> Used in masked language modeling (MLM)  

- Model predicts the masked word using context  

Example: <br>
I love [MASK] life


Used in:
- Pretraining models like BERT  

---

### Task-Specific Tokens

> Custom tokens introduced for specific tasks  

Examples: <br>
[SOURCE] Hello [TARGET] Bonjour


- Helps model distinguish different roles in input  
- Common in:
  - Machine translation  
  - Instruction tuning  
  - Multi-task learning  

---