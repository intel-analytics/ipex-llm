# Efficient Multi-Task Learning via Generalist Recommender (GRec)

Implementation of Generalist Recommender (GRec), an end-to-end efficient and scalable recommender system
designed to train a single model that could generalize across multiple search & recommender tasks
while at the same time being highly efficient.

This work was contributed by the **_Verizon AI Center team_**.

**GRec** has the following highlights:
- **Input Modalities** - Designed to handle inputs of multi-modalities (including categorical and numerical data, texts, and images) by utilizing NLP heads, parallel Transformers, as well as wide and deep in the model architecture.
- **Highly Efficient** - Adopts a newly proposed task-sentence level routing mechanism to scale the model capabilities on multiple tasks without compromising performance.
- **Production Ready** - Successfully deployed at Verizon and demonstrates significant performance improvement over the baseline in both offline and online settings.
