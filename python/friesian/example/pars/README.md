# Pathways Recommender Service (ParS)

Implementation of Pathways Recommender Service (PaRS) which is a multi-task recommender service designed to train a single model that could generalize across multiple search & recommender tasks while being highly efficient. This work was contributed by the Verizon AI Center team.

PaRS was designed with the following values in mind:
- **Unification** - Users should be able to consolidate multiple existing recommender models into one using PaRS
- **Highly Efficient** - Users should expect to gain efficiency across model training, deployment, inference, and production maintenance / monitoring
- **Platform Agnostic** - Users should be able to train and inference PaRS on both CPU / GPU enabled computation platforms