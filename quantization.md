Here’s a brief step‑by‑step look at how quantization usually works in transformer models:

### 1. Start with FP32 (or BF16) weights and activations
- Models are normally trained and stored in **FP32** (32‑bit floating point).  
- Some newer setups use **BF16** (16‑bit floating point) for efficiency, but it’s still higher precision than INT8.

### 2. Quantize the weights
- Each weight (a parameter in the model) is converted from FP32/BF16 into **INT8**.  
- This is done by mapping the floating‑point values into a smaller integer range (−128 to +127).  
- A scaling factor is stored so the model can “interpret” the integers correctly.

### 3. Quantize the activations
- Activations (the outputs of each layer during inference) are also quantized to INT8.  
- This step is trickier because activations vary depending on the input data.  
- Often, dynamic quantization is used: the scaling factor is recalculated on the fly for each batch of inputs.

### 4. Run inference in INT8
- With both weights and activations in INT8, the model can do matrix multiplications much faster and with less memory.  
- Hardware like GPUs, TPUs, or specialized accelerators are optimized for these low‑precision operations.

### 5. Dequantize when needed
- At certain points (e.g., final output layer), results may be converted back to FP32 for accuracy.  
- This ensures the predictions are interpretable and consistent with expected formats.

---

🔑 **Key idea:**  
Quantization shrinks FP32/BF16 values into INT8 integers using scaling factors. The challenge is making sure the rounding errors don’t pile up across dozens of transformer layers. That’s why the report you’re reading highlights the risk of “compounding errors.”

let’s walk through a **tiny numerical example** of quantization so you can see how FP32 values get mapped into INT8:

---

### Step 1: Original FP32 value
Suppose a weight in the model is:

\[
w = 2.75 \quad \text{(stored in FP32)}
\]

---

### Step 2: Define the INT8 range
- INT8 can only store integers between **−128 and +127**.  
- To fit FP32 values into this range, we use a **scale factor**.

Let’s say the largest weight in this layer is about **5.0**.  
We choose a scale factor:

\[
\text{scale} = \frac{5.0}{127} \approx 0.039
\]

---

### Step 3: Quantize (map FP32 → INT8)
We divide the FP32 value by the scale and round:

\[
q = \text{round}\left(\frac{2.75}{0.039}\right) \approx \text{round}(70.5) = 71
\]

So the FP32 weight **2.75** becomes INT8 value **71**.

---

### Step 4: Dequantize (map INT8 → FP32 when needed)
To recover an approximate FP32 value:

\[
w' = q \cdot \text{scale} = 71 \cdot 0.039 \approx 2.77
\]

This is very close to the original **2.75**, but not exact — that’s the rounding error.

---

### Why this matters
- Each layer introduces tiny differences like this.  
- Across **32–50 transformer layers**, those small errors can accumulate.  
- Usually the errors stay small enough that accuracy is fine, but in edge cases (like COPA dataset in your report), they can cause noticeable fragility.

---

So in short:  
- FP32 values are shrunk into INT8 using a scale factor.  
- INT8 saves memory and speeds up inference.  
- Dequantization brings them back close to the original, but not perfectly.  

Would you like me to also show you a **visual analogy** (like compressing a photo into fewer colors) to make the idea of quantization even more intuitive?
