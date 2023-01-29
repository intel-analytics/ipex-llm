Cluster Serving
=========================

BigDL Cluster Serving is a lightweight distributed, real-time serving solution that supports a wide range of deep learning models (such as TensorFlow, PyTorch, Caffe, BigDL and OpenVINO models). It provides a simple pub/sub API, so that the users can easily send their inference requests to the input queue (using a simple Python API); Cluster Serving will then automatically manage the scale-out and real-time model inference across a large cluster (using distributed streaming frameworks such as Apache Spark Streaming, Apache Flink, etc.)

----------------------



.. grid:: 1 2 2 2
    :gutter: 2

    .. grid-item-card::

        **Get Started**
        ^^^

        Documents in these sections helps you getting started quickly with Serving.

        +++

        :bdg-link:`Serving in 5 minutes <./QuickStart/serving-quickstart.html>` |
        :bdg-link:`Installation <./ProgrammingGuide/serving-installation.html>`

    .. grid-item-card::

        **Key Features Guide**
        ^^^

        Each guide in this section provides you with in-depth information, concepts and knowledges about DLLib key features.

        +++

        :bdg-link:`Start Serving <./ProgrammingGuide/serving-start.html>` |
        :bdg-link:`Inference <./ProgrammingGuide/serving-inference.html>`


    .. grid-item-card::

        **Examples**
        ^^^

        Cluster Serving Examples and Tutorials.

        +++

        :bdg-link:`Examples <./Example/example.html>`

    .. grid-item-card::

        **MISC**
        ^^^

        Cluster Serving

        +++

        :bdg-link:`FAQ <./FAQ/faq.html>` |
        :bdg-link:`Contribute <./FAQ/contribute-guide.html>`



..  toctree::
    :hidden:

    Cluster Serving Document <self>