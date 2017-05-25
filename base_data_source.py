"""
Base class to be inherited by data sources. Offers the interface any data source
should provide.
"""
class BaseDataSource:
    def __init__(self):
        raise NotImplementedError("Please implement this method")


    """
    Loads pretrained word embeddings from a file and creates an embedding matrix
    with one one embedding (i.e. row) for each word in the vocabulary.
    """
    def get_embeddings(self, vocab):
        pass

    """
    Returns a tuple (images, labels), where images is a list of images and labels
    is a list of labels. If num_samples is provided, it will only retrieve that
    many samples (i.e. #{images} = #{labels} = num_samples).
    """
    def train(self, num_samples=None):
        raise NotImplementedError("Please implement this method")

    """
    Returns a tuple (images, labels), where images is a list of images and labels
    is a list of labels. If num_samples is provided, it will only retrieve that
    many samples (i.e. #{images} = #{labels} = num_samples).
    """
    def validation(self, num_samples=None):
        raise NotImplementedError("Please implement this method")

    """
    Returns a tuple (images, labels), where images is a list of images and labels
    is a list of labels. If num_samples is provided, it will only retrieve that
    many samples (i.e. #{images} = #{labels} = num_samples).
    """
    def test(self, num_samples=None):
        raise NotImplementedError("Please implement this method")
