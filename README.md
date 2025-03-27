# ResNet18 Optimization Journey with Cifar10: Towards Efficient Deep Learning  

This project explores various optimization techniques to improve ResNet18's efficiency on the CIFAR-10 dataset while maintaining high accuracy. The objective is to achieve a **90% accuracy** while minimizing a **custom efficiency score** that considers model size, computation cost, and pruning levels.

---

## Preliminary Study  
Before implementing any optimization technique, we analyzed the research paper:  
**"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"**  
by Mingxing Tan and Quoc V. Le (ICML 2019).  

This study introduced an optimized model scaling approach, balancing **depth, width, and resolution**, which inspired our approach to improving ResNet18.

---

## Objective  
The goal is to train an optimized ResNet18 that **achieves 90% accuracy on CIFAR-10** while minimizing the following **efficiency score**:

$$
\text{score} = \frac{(1 - (p_s + p_u)) \cdot \frac{q_w}{32} \cdot w}{5.6 \times 10^6} + \frac{(1 - p_s) \cdot \frac{\max(q_w, q_a)}{32} \cdot f}{2.8 \times 10^8}
$$

Where:  
- \( $p_s$ \): structured pruning ratio  
- \( $p_u$ \): unstructured pruning ratio  
- \( $q_w$ \): weight quantization  
- \( $q_a$ \): activation quantization  
- \( w \): number of weights  
- \( f \): number of multiply-add operations (MACs)  
- **Reference values:**  
  - \( $5.6 \times 10^6$ \) parameters (ResNet18 baseline)  
  - \( $2.8 \times 10^8$ \) MAC operations (ResNet18 in half precision)  

The lower the score, the more efficient the model.

---

## Optimization Techniques  

We applied several optimization strategies and their combinations to reduce model size and computational cost while preserving accuracy:

1. **Pruning**  
   - Structured pruning (entire channels or layers removed)  
   - Unstructured pruning (removal of individual weights)  

2. **Quantization**  
   - Reducing weight and activation precision (e.g., INT8 instead of FP32)  

3. **Factorization**  
   - Depthwise separable convolutions  
   - Grouped convolutions (tested with groups = 2 and groups = 8)  

4. **Knowledge Distillation**  
   - Training a smaller model (student) using a larger, well-trained model (teacher)  

5. **Ensemble Method**  
   - Combining multiple optimized models in a voting-based ensemble  

---

## Results & Trade-offs  

After testing multiple configurations, we identified models that **achieve a balance between accuracy and efficiency**.  
The ensemble method further improved robustness by leveraging diverse model architectures.

📊 **Best model trade-off:**  
- **Accuracy:** ~93.47%  
- **Score:** 0.597288

---

