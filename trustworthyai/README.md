# Trustworthy AI

## Guiding Principles

### **Privacy**

AI should comply with privacy laws and regulations and meet societal norms for personal data and information privacy.

### **Safety and Security**

Ensure that AI systems perform as intended and avoid unintended harm and malicious threats.

### **Transparency**

Make AI technology understandable to people. Explain, in non-technical language, how an AI system arrived at its output.

### **Nondiscrimination**

Minimize bias in our AI systems and give all groups an equal opportunity to benefit from AI.

### Accountability

Accountability **refers to the idea that artificial intelligence should be developed, deployed, and utilized such that responsibility for bad outcomes can be assigned to liable parties**.

## Bias mitigation

### **Data Collection Diversification**

Ensuring that the training data is representative of diverse groups and experiences. For example, if you’re training an image recognition model, it should include diverse demographics, including race, gender, and age groups.

### **Data Preprocessing**

Before training, techniques can be applied to balance the data by oversampling underrepresented groups or undersampling overrepresented ones. This helps create a more balanced dataset and reduces the risk of the model favoring the majority group.

### **Synthetic Data Generation**

In cases where data is sparse for certain groups, synthetic data can be generated using methods like data augmentation (e.g. SMOTE) or generative models (e.g., GANs) to balance the representation of various groups. 

### Fairness Constraints

Adding fairness constraints during model training that limit the model’s ability to exploit certain biases in the data. These constraints can be designed to enforce fairness by ensuring equal treatment or outcomes across different demographic groups.

### **Regularization Techniques**

These methods aim to prevent the model from overfitting to biased patterns in the data. Regularization can be applied to reduce the impact of sensitive features on the model’s predictions.

### **Bias Audits**

External or internal audits of AI models can assess how well a system adheres to fairness principles and legal requirements.

## Algorithmic Fairness Metrics

### **Demographic Parity**

This measures whether different groups (e.g., gender, race) receive equal positive outcomes (e.g., loan approvals, hiring decisions). If one group is favored disproportionately, the model is considered biased.

### **Equalized Odds**

This metric requires that the model has equal true positive rates and equal false positive rates across different groups, ensuring that all groups are treated similarly with respect to the model’s decisions.

## NeMo Guardrails

NeMo Guardrails is an open-source toolkit for easily adding *programmable guardrails* to LLM-based conversational applications. Guardrails (or "rails" for short) are specific ways of controlling the output of a large language model, such as not talking about politics, responding in a particular way to specific user requests, following a predefined dialog path, using a particular language style, extracting structured data, and more.

## Human-in-the-loop

A **human-in-the-loop** (or "on-the-loop") workflow integrates human input into automated processes, allowing for decisions, validation, or corrections at key stages. This is especially useful in **LLM-based applications**, where the underlying model may generate occasional inaccuracies. In low-error-tolerance scenarios like compliance, decision-making, or content generation, human involvement ensures reliability by enabling review, correction, or override of model outputs.