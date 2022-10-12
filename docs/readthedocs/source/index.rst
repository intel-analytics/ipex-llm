BigDL Documentation
===========================

BigDL seamlessly scales your data analytics & AI applications from laptop to cloud, with the following libraries:

- `Orca <doc/Orca/index.html>`_: Distributed Big Data & AI (TF & PyTorch) Pipeline on Spark and Ray
- `Nano <doc/Nano/index.html>`_: Transparent Acceleration of Tensorflow & PyTorch Programs
- `DLlib <doc/DLlib/index.html>`_ "Equivalent of Spark MLlib" for Deep Learning
- `Chronos <doc/Chronos/index.html>`_: Scalable Time Series Analysis using AutoML
- `Friesian <doc/Friesian/index.html>`_: End-to-End Recommendation Systems
- `PPML <doc/PPML/index.html>`_ (experimental): Secure Big Data and AI (with SGX Hardware Security)

------

Choosing the right BigDL library
---------------------------

.. graphviz::

    digraph BigDLDecisionTree {
        node [color="#459db9" shape="box" fontname="Arial" fontsize=14]
        edge [fontname="Arial" fontsize=12]
        
        Feature1 [label="Hardware Secured Big Data & AI?"]
        Feature2 [label="Python vs. Scala/Java?"]
        Feature3 [label="What type of application?"]
        Feature4 [label="Domain?"]
        
        Orca[href="../doc/Orca/index.html" target="_blank" target="_blank" style="rounded,filled" fillcolor="#459db9" fontcolor="#ffffff"]
        Nano[href="../doc/Nano/index.html" target="_blank" target="_blank" style="rounded,filled" fillcolor="#459db9" fontcolor="#ffffff"]
        DLlib1[label="DLlib" href="../doc/DLlib/index.html" target="_blank" style="rounded,filled" fillcolor="#459db9" fontcolor="#ffffff"]
        DLlib2[label="DLlib" href="../doc/DLlib/index.html" target="_blank" style="rounded,filled" fillcolor="#459db9" fontcolor="#ffffff"]
        Chronos[href="../doc/Chronos/index.html" target="_blank" style="rounded,filled" fillcolor="#459db9" fontcolor="#ffffff"]
        Friesian[href="../doc/Friesian/index.html" target="_blank" style="rounded,filled" fillcolor="#459db9" fontcolor="#ffffff"]
        PPML[href="../doc/PPML/index.html" target="_blank" style="rounded,filled" fillcolor="#459db9" fontcolor="#ffffff"]
        
        Feature1 -> Feature2[label="No"]
        Feature1 -> PPML[label="Yes"]
        
        Feature2 -> Feature3[label="Python"]
        Feature2 -> DLlib1[label="Scala/Java"]
        
        Feature3 -> Orca[label="Distributed Big Data \n + \n  AI (TF/PyTorch)"]
        Feature3 -> Nano[label="Accelerate \n TensorFlow / PyTorch"]
        Feature3 -> DLlib2[label="DL for Spark MLlib"]
        Feature3 -> Feature4[label="High Level \n  App Framework"]
        
        Feature4 -> Chronos[label="Time Series"]
        Feature4 -> Friesian[label="Recommender System"]
    }