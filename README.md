### **README - Toxicity Prediction Using Graph Neural Networks (GNNs)**  

---

## **Project Title**  
**Toxicity Prediction Using Graph Neural Networks (GNNs)**  
Leveraging deep learning techniques such as Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs) to improve toxicity prediction in chemical molecules.

---

## **Project Overview**  
This project investigates the use of Graph Neural Networks for predicting toxicity endpoints in chemical compounds. We compare various models, including Naïve Bayes, GCNs (5-node input), GATs, and Morgan fingerprint-enhanced GCN models to understand their performance in toxicity prediction. The goal is to provide a robust framework that efficiently predicts chemical toxicity and contributes to improved drug discovery while reducing reliance on traditional experimental methods.

---

## **Objectives**  
- Develop a GNN-based framework for predicting toxicity endpoints.  
- Improve model performance by integrating Morgan fingerprints for enhanced feature representation.  
- Compare GNN models with traditional ML classifiers such as Naïve Bayes.  
- Optimize model performance using data balancing techniques and improved architecture.  
- Provide interpretability solutions to improve the transparency of deep learning predictions.  

---

## **Dataset Used**  
This study uses the **Tox21 dataset**, a publicly available dataset from **NCBI PubChem**. It contains chemical structures represented as SMILES strings or CAS IDs, along with their respective toxicity endpoints.

### **Key Dataset Attributes**  
- **Molecular fingerprints:** Morgan fingerprints and Extended Connectivity Fingerprints (ECFP)  
- **Molecular descriptors:** Atomic properties, bond types, degree, valence  
- **Toxicity Endpoints:**
  - **NR-AR:** Nuclear Receptor - Androgen Receptor  
  - **SR-ARE:** Stress Response - Antioxidant Response Element  
  - **NR-Aromatase:** Nuclear Receptor - Aromatase  
  - **NR-ER-LBD:** Nuclear Receptor - Estrogen Receptor Ligand Binding Domain  
  - **NR-AhR:** Nuclear Receptor - Aryl Hydrocarbon Receptor  
  - **SR-MMP:** Stress Response - Mitochondrial Membrane Potential  
  - **NR-ER:** Nuclear Receptor - Estrogen Receptor  
  - **NR-PPAR-gamma:** Nuclear Receptor - Peroxisome Proliferator-Activated Receptor Gamma  
  - **SR-p53:** Stress Response - p53 Tumor Suppressor  
  - **SR-ATAD5:** Stress Response - ATPase Family AAA Domain-Containing Protein 5  
  - **NR-AR-LBD:** Nuclear Receptor - Androgen Receptor Ligand Binding Domain  

---

## **Pre-processing Steps**  
1. **Data Cleaning:**  
   - Removed invalid or corrupted molecules.  
2. **Molecular Representation:**  
   - Converted molecules into graph structures where nodes represent atoms and edges represent bonds.  
3. **Feature Engineering:**  
   - Generated Morgan fingerprints (radius=2, 1024-bit) as node attributes.  
4. **Graph Construction:**  
   - Transformed data into PyTorch Geometric format, defining edge connectivity and node feature matrices.  

---

## **Model Implementation**  

### **1. Traditional ML Algorithms**  
- **Naïve Bayes Classifier:**  
   - Utilized as a baseline classification model based on molecular descriptors like FW, DSSTox_CID, and Molecular Formula.  
- **Random Forest Classifier:**  
   - Trained on Morgan fingerprints to capture complex chemical interactions.  

---

### **2. Graph Neural Networks (GNNs)**  
**GCN (5-node input)**  
- **Input:** Molecular graphs with node features as Morgan fingerprints and edge connectivity from bond information.  
- **Layers:**  
   - **GCNConv Layer 1:** Extracts local node interactions.  
   - **ReLU Activation & Batch Normalization:** Ensures stability.  
   - **GCNConv Layer 2:** Captures deeper chemical dependencies.  
   - **Global Mean Pooling:** Aggregates node-level embeddings into graph-level representation.  
   - **Fully Connected Layer:** Maps final embeddings to toxicity classifications.  

**GAT (Graph Attention Network)**  
- **Self-Attention Mechanism:** Assigns higher weights to critical atomic interactions.  
- **Architecture:**  
   - Input Layer → GATConv Layer 1 → Batch Normalization → ReLU Activation  
   - GATConv Layer 2 → Batch Normalization → ReLU Activation  
   - Global Mean Pooling → Fully Connected Layer → Classification Output  

---

### **3. Data Balancing Techniques**  
To address class imbalance, an undersampling strategy was applied where:  
- The majority class (non-toxic molecules) was reduced to match the minority class (toxic molecules).  
- The balanced dataset was shuffled to ensure fair training conditions.  

---

## **Results and Discussion**  
The study evaluated the impact of various GNN models and molecular representations:  
- The **Morgan fingerprint-based GCN** model outperformed the traditional 5-node input GCN model with improved recall and precision for both toxic and non-toxic classes.  
- Morgan fingerprints extracted higher-order molecular patterns, leading to more balanced predictions.  
- GNNs demonstrated robust scalability and adaptability in predicting toxicity across different endpoints.  
- The **Naïve Bayes** classifier showed poor performance, underscoring the limitations of probabilistic models for toxicity prediction.  
- The **GAT Model** demonstrated strong performance with improved stability and better differentiation of complex molecular interactions.

---

## **Future Works**  
- **Advanced GNN Architectures:** Explore GATs, Transformer-based models, and GINs for improved interpretability.  
- **Semi-Supervised Learning:** Adopt self-training and contrastive learning to improve performance using unlabeled data.  
- **Transfer Learning:** Leverage pretrained GNNs for improved performance in low-data toxicity prediction.  
- **Multi-Task Learning:** Develop models that predict multiple toxicity endpoints simultaneously to improve model generalization.  
- **Explainability and Interpretability:** Implement attention mechanisms and SHAP values for improved model transparency.  

---

## **Authors**  
- **Samrat Raj Sharma**  
- **Sanjan TP Gupta**

---

## **Acknowledgments**  
We acknowledge the contributions of research papers and open-source resources that guided this project. Special thanks to the Tox21 dataset providers and PyTorch Geometric community.

---

## **Contact**  
For queries, reach out at: **Samratsharma979@gmail.com**
