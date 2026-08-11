# A CNN–SNN Framework for Originality Assessment and Style Attribution of Traditional Fine Arts Artworks at EARIST Manila

This study addresses subjectivity in Fine Arts assessment by proposing a dual-stage Convolutional Neural Network–Siamese Neural Network (CNN-SNN) framework for objective style attribution and originality evaluation. The framework evaluates stylistic features across two distinct sources: a primary dataset of 102 local student portfolios and a secondary global dataset of 50 master artists from WikiArt. Analysis is conducted through three specialized pipelines: Color, Brushstroke, and Texture. EfficientNet-B3 achieved up to 88.81% accuracy on the Global dataset and 92.75% on the Local dataset. An originality analysis revealed that 73.5% of students were classified as "Very Similar" and 26.5% as "Similar," with an originality range of 28 points (4.87%–33.01%). The framework provides quantitative similarity measures and percentile-based originality scores, supporting objective evaluation of artistic growth, stylistic influence, and academic integrity in Fine Arts education.

[![Tech Stack](https://img.shields.io/badge/Python-HTML-CSS-blue?style=for-the-badge)]()

---

## 📸 System Previews

<table align="center">
  <tr>
    <td align="center"><b>Portfolios of Other Artists</b></td>
    <td align="center"><b>Sample Portfolio Analysis of a Single Artist</b></td>
  </tr>
  <tr>
    <td align="center">
      <img width="500" alt="Portfolios of Other Artists" src="https://github.com/user-attachments/assets/86ed9857-09dd-4d3c-a6c5-7bf5b8c439f7" />
    </td>
    <td align="center">
      <img width="500" alt="Sample Portfolio Analysis of a Single Artist" src="https://github.com/user-attachments/assets/898a3fc3-5f23-4a4f-b337-043b8436cefb" />
    </td>
  </tr>
</table>

---

## 💡 Key Highlights & Findings
* **Dual-Stage Architecture:** Combines CNN and Siamese Neural Network (CNN-SNN) models to handle objective style attribution and originality evaluation.
* **Specialized Pipelines:** Analyzes stylistic features through three distinct domains: **Color**, **Brushstroke**, and **Texture**.
* **Comprehensive Datasets:** Evaluates across a primary dataset of 102 local student portfolios and a secondary global dataset of 50 master artists from WikiArt.
* **High Performance:** EfficientNet-B3 achieved up to **88.81% accuracy** on the Global dataset and **92.75% accuracy** on the Local dataset.
* **Originality Insights:** Found that 73.5% of students fell into the "Very Similar" classification and 26.5% into "Similar," displaying an originality range of 28 points (4.87%–33.01%).

---

## 🛠️ Tech Stack & Implementation
* **Language & Logic:** Python (Model training, pipelines, and backend processing)
* **Interface & Presentation:** HTML, CSS (Web-based deployment and visualization interface)
