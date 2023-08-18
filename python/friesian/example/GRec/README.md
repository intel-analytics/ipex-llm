# Efficient Multi-Task Learning via Generalist Recommender (GRec)

Implementation of Generalist Recommender (GRec) which is an end-to-end efficient and scalable recommender system
designed to train a single model that could generalize across multiple search & recommender tasks
while at the same time being highly efficient. This work was contributed by the Verizon AI Center team.

**GRec** has the following highlights:
- **Input Modalities** - Designed to take inputs of multi-modalities (including categorical and numerical data, texts, and images) by utilizing NLP heads, parallel Transformers, as well as wide and deep in the model architecture.
- **Highly Efficient** - Adopts a newly proposed task-sentence level routing mechanism to scale the model capabilities on multiple tasks without compromising performance.
- **Production Ready** - Significant performance improvement over the baseline in both offline and online A/B testing settings at Verizon.
