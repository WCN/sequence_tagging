from model.data_utils import AnnotationDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = AnnotationDataset(
        '/ben/textkernel/annotations', file_extension='ann', processing_word=processing_word
    )
    train = AnnotationDataset(
        '/u02/textkernel/annotations', file_extension='ann', processing_word=processing_word
    )

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
