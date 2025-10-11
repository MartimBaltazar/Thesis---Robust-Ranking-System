# A Robust and Ethical Group Ranking System and Its Resistance to Bribery

This repository contains the code and experimental results for the Master of Science thesis, "**A Robust and Ethical Group Ranking System and Its Resistance to Bribery**," submitted by **Martim Nunes Silva Baltazar** to obtain the Master of Science Degree in Computer Science and Engineering.

The core of this work is the development and evaluation of a **User-Agnostic Multipartite Ranking System (UAMRS)** that is designed to comply with ethical AI regulations (such as the EU AI Act's prohibition on individual user scoring) while maintaining robustness against adversarial attacks and manipulation.

---

## 🚀 Project Goal

To develop a ranking system that aligns with today's ethical and technical standards by focusing on item-based characteristics, ensuring **AI Act compliance**, **fairness**, and strong **resistance to bribery and spam**.

---

## 💡 Key Contributions

1.  **User-Agnostic Multipartite Framework (UAMRS):** An extension of the user-agnostic bipartite framework into a multipartite setting, grouping users based on rating patterns to enhance robustness and confine the impact of individual users to their cluster.
2.  **Complementary Cluster Concept:** A novel mechanism to manage clusters that are too small, aggregating users below a predefined size threshold into a complementary cluster to ensure meaningful results.
3.  **Empirical Superiority:** Systematic evaluation demonstrating that the proposed UAMRS framework generally outperforms four state-of-the-art ranking approaches in terms of **bribery resistance**, and achieves a significant reduction (up to a factor of 100) in **demographic bias** compared to a user-agnostic bipartite approach.

---

## 💻 Methodology: User-Agnostic Multipartite Ranking System (UAMRS)

The UAMRS operates through the following key steps:

1.  **User Clustering:** Users are partitioned into disjoint clusters based on their rating patterns. The **Kolmogorov similarity measure** was used for this purpose, with a similarity threshold of **0.99** for edge creation.
2.  **Rating Adjustment and Convergence:** For each item within each cluster, the algorithm iteratively filters out ratings that deviate significantly from the cluster mean. Specifically, users whose ratings deviate **more than one standard deviation** from the mean are discarded, and the process repeats until a fixed point (convergence) is reached.
3.  **Final Ranking:** Final item rankings are aggregated as a weighted average of the cluster-wise ranks, where each cluster contributes according to its size.

This approach avoids assigning individual reputation scores, thus maintaining compliance with ethical guidelines.

---

## 📊 Experimental Evaluation

The proposed system was evaluated against four state-of-the-art ranking systems (Aggregated Average RS, Reputation-based Bipartite RS, User-Agnostic Bipartite RS, and Reputation-based Multipartite RS).

### Datasets Used
| Dataset | Type | Item Count ($\vert I \vert$) | User Count ($\vert U \vert$) | Ratings Count ($\vert R \vert$) |
| :--- | :--- | :--- | :--- | :--- |
| **BookCrossing** | Book Ratings | | | |
| **Movielens-1M** | Movie Ratings | | | |
| **Amazon Luxury** | Product Ratings | | | |

*Note: Datasets were preprocessed to ensure a 5-core version for analysis.*

### Evaluation Metrics

| Metric | Purpose | Key Finding for UAMRS |
| :--- | :--- | :--- |
| **Effectiveness** (Spearman's $\tau$) | Measures how well the UAMRS's ranking correlates with the ground truth (Aggregated Average). | Achieved rankings that reflect underlying user preferences. |
| **Fairness** (DDRB) | Measures ranking bias across demographic groups (e.g., age, sex). | Reduced demographic bias by a factor of 100 compared to user-agnostic bipartite approach. |
| **Robustness** (Kendall's $\tau$) | Measures resistance to random spamming (noise). | Exhibited enhanced spam resistance on two of the three datasets. |
| **Bribery Resistance** ($\pi_{\sigma_i}$, $\Delta J_i$) | Measures the profitability of manipulation attacks (Push/Nuke). | Showed substantial improvements, with profitable attacks in only 7 of 18 scenarios, a better result than state-of-the-art baselines. |

---

## 🛠️ Reproduction and Usage

The entire codebase for the ranking systems and experimental setup is organized within the single `scripts/` directory, structured by the dataset used.

### Key Directories

* `/scripts`: The root folder containing all source code and experimental files.
    * `/scripts/{dataset_name}`: Top-level directory for each dataset used in the thesis (e.g., `amazonLuxury`, `bookcrossing`, `ml-100k`, `ml-1m`).
        * `/demographics`: Contains scripts (`.py` files) used for processing demographic data and conducting the **Fairness** experiments (DDRB metric).
        * `/preProcessing`: Contains scripts for preparing the datasets, including creating simulated spammer or attacker ratings for the **Robustness** and **Bribery Resistance** experiments.
        * `/rankings/py`: The core Python implementations of all five ranking algorithms (AA, RBRS, UABRS, RMRS, UAMRS) tailored for the specific dataset.
        * `/rankings/ipynb`: Jupyter notebooks (`.ipynb` files) that run the ranking algorithms and perform the final statistical analysis and visualization for the **Effectiveness**, **Robustness**, and **Bribery Resistance** evaluations.

### How to Run Experiments

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/MartimBaltazar/Thesis](https://github.com/MartimBaltazar/Thesis)—Robust-Ranking-System.git
    cd Thesis—Robust-Ranking-System
    ```
2.  **Install Dependencies:** (A `requirements.txt` file detailing necessary libraries would be placed here.)
    ```bash
    # Example command
    pip install -r requirements.txt
    ```
3.  **Run Main Evaluation Script:** The experiments are typically run via the Jupyter notebooks located in the `/scripts/{dataset_name}/rankings/ipynb` folders.
    ```bash
    # Example command to run the UAMRS model for MovieLens-1M
    jupyter notebook scripts/ml-1m/rankings/ipynb/userAgnosticMultipartite_ml-1m.ipynb
    ```

---

## 🎓 Citation

If you use this work, please cite the original thesis:
@mastersthesis{baltazar2025robust,
title={A Robust and Ethical Group Ranking System and Its Resistance to Bribery},
author={Baltazar, Martim Nunes Silva},
year={2025},
school={Instituto Superior Técnico, Universidade de Lisboa},
supervisor={Ramos, Guilherme}
}

---

## 📧 Contact
For any questions or suggestions, please contact the author or supervisor:
* **Author**: Martim Nunes Silva Baltazar (martimnunesbaltazar@gmail.com)
* **Supervisor**: Guilherme Ramos (guilherme.ramos@tecnico.ulisboa.pt)
