from keras.callbacks import Callback
from sqlalchemy import (create_engine,
        Column,
        Integer,
        String,
        Boolean,
        ForeignKey,
        DateTime,
        )
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import Engine
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import ENUM, ARRAY
from sqlalchemy import event
from sklearn.metrics import confusion_matrix
import numpy as np
import datetime

Base = declarative_base()
Session = sessionmaker()


class RunConfiguration(Base):
    __tablename__ = 'run_configurations'

    run_id = Column(Integer, primary_key=True, autoincrement=False)
    training_good_files = Column(ARRAY(String), nullable=False)
    training_defect_files = Column(ARRAY(String), nullable=False)
    validation_good_files = Column(ARRAY(String), nullable=False)
    validation_defect_files = Column(ARRAY(String), nullable=False)
    test_good_files = Column(ARRAY(String), nullable=False)
    test_defect_files = Column(ARRAY(String), nullable=False)
    serialized_model = Column(String, nullable=False)
    grayscale = Column(Boolean, nullable=False)
    undersampling = Column(Boolean, nullable=False)

    epochs = relationship('EpochStats', back_populates='run')

class EpochStats(Base):
    __tablename__ = 'epoch_stats'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('run_configurations.run_id'))
    epoch_id = Column(Integer, nullable=False)
    datetime = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    true_positive = Column(Integer, nullable=False)
    true_negative = Column(Integer, nullable=False)
    false_positive = Column(Integer, nullable=False)
    false_negative = Column(Integer, nullable=False)
    data_nature = Column(ENUM('training', 'test', 'validation', name='DataNature'), nullable=False)

    run = relationship('RunConfiguration', back_populates='epochs')

    __table_args__ = (
            UniqueConstraint('run_id', 'epoch_id', 'data_nature', name='run_epoch_nature'),
            )


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
