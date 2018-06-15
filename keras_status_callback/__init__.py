from keras.callbacks import Callback
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix

from .models import Base, RunConfiguration, EpochStats

Session = sessionmaker()


class DatasetDescription(object):
    def __init__(self, name, X, y, good_files, defect_files):
        self.name = name
        self.X = X
        self.y = y
        self.good_files = good_files
        self.defect_files = defect_files


class StatusCallback(Callback):

    ALLOWED_NAMES = {'training', 'test', 'validation'}

    def __init__(self, run_id, db_connection_string, undersampling, grayscale,
            verbose=False, reset=False):
        self.engine = create_engine(db_connection_string, echo=verbose)
        if reset:
            Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        Session.configure(bind=self.engine)
        self.session = Session()

        self.run_id = run_id
        self.training = None
        self.test = None
        self.validation = None
        self.grayscale = grayscale
        self.undersampling = undersampling

    def set_data(self,
            name,
            X, y,
            good_files, defect_files):

        if name not in self.ALLOWED_NAMES:
            raise ValueError('Dataset name not in allowed names: {}'.format(self.ALLOWED_NAMES))

        dataset= DatasetDescription(
                name=name,
                X=X,
                y=y,
                good_files=good_files,
                defect_files=defect_files)

        if name == 'training':
            self.training = dataset
        elif name == 'test':
            self.test = dataset
        elif name == 'validation':
            self.validation = dataset

    @property
    def data_set(self):
        return self.training is not None or self.test is not None or self.validation is not None

    def on_train_begin(self, logs=None):
        if not self.data_set:
            raise ValueError('Initial run data not set. Please call `set_data` for each training/test/validation dataset first')

        self.run_config = RunConfiguration(
                run_id=self.run_id,
                training_good_files = self.training.good_files,
                training_defect_files = self.training.defect_files,
                validation_good_files = self.validation.good_files if self.validation is not None and self.validation.good_files is not None else [],
                validation_defect_files = self.validation.defect_files if self.validation is not None and self.validation.defect_files is not None else [],
                test_good_files = self.test.good_files if self.test is not None and self.test.good_files is not None else [],
                test_defect_files = self.test.defect_files if self.test is not None and self.test.defect_files is not None else [],
                serialized_model = self.model.to_json(),
                grayscale = self.grayscale,
                undersampling = self.undersampling,
                )
        self.session.add(self.run_config)
        self.session.commit()

    def on_epoch_end(self, epoch, logs=None):
        for data_nature in ['training', 'test', 'validation']:
            collection = getattr(self, data_nature, None)
            if collection is None:
                continue

            X = collection.X
            y = collection.y

            if X is None or y is None:
                continue

            y_pred = self.model.predict(X)
            y_true, y_pred = self._format_y(y), self._format_y(y_pred)
            mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
            tn, fp, fn, tp = list(map(int, mat.ravel()))

            epoch_stats = EpochStats(
                    run_id=self.run_id,
                    data_nature=data_nature,
                    epoch_id=epoch,
                    true_positive = tp,
                    true_negative = tn,
                    false_positive = fp,
                    false_negative = fn,
                    )
            self.session.add(epoch_stats)

        self.session.commit()

    def _compute_stats(self, X, y):
        return None


    def _format_y(self, y):
        if self._is_one_hot(y):
            y = from_categorical(y)

        assert len(y.shape) == 1
        return np.round(y).astype(int)

    def _is_one_hot(self, y):
        return len(y.shape) > 1


def from_categorical(y):
    count = y.shape[1]

    out = np.zeros(y.shape[0])
    for i in range(count):
        idx = y[:, i] == 1
        out[idx] = i

    return out
