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
from sqlalchemy.dialects.postgresql import ARRAY
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

    run = relationship('RunConfiguration', back_populates='epochs')

    __table_args__ = (
            UniqueConstraint('run_id', 'epoch_id', name='run_epoch'),
            )


class StatusCallback(Callback):
    def __init__(self, run_id, db_connection_string, verbose=False, reset=False):
        self.engine = create_engine(db_connection_string, echo=verbose)
        if reset:
            Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        Session.configure(bind=self.engine)
        self.session = Session()

        self.run_id = run_id
        self.X = None
        self.y = None
        self.training_good_files = None
        self.training_defect_files = None
        self.validation_good_files = None
        self.validation_defect_files = None
        self.test_good_files = None
        self.test_defect_files = None
        self.grayscale = None
        self.undersampling = None

        self.data_set = False

    def set_data(self,
            X, y,
            training_good_files,
            training_defect_files,
            validation_good_files,
            validation_defect_files,
            test_good_files,
            test_defect_files,
            grayscale,
            undersampling):

        self.X = X
        self.y = y

        self.training_good_files = training_good_files
        self.training_defect_files = training_defect_files
        self.validation_good_files = validation_good_files
        self.validation_defect_files = validation_defect_files
        self.test_good_files = test_good_files
        self.test_defect_files = test_defect_files
        self.grayscale = grayscale
        self.undersampling = undersampling
        self.data_set = True

    def on_train_begin(self, logs=None):
        if not self.data_set:
            raise ValueError('Initial run data not set. Please call `set_data` first')

        self.run_config = RunConfiguration(
                run_id=self.run_id,
                training_good_files = self.training_good_files,
                training_defect_files = self.training_defect_files,
                validation_good_files = self.validation_good_files,
                validation_defect_files = self.validation_defect_files,
                test_good_files = self.test_good_files,
                test_defect_files = self.test_defect_files,
                serialized_model = self.model.to_json(),
                grayscale = self.grayscale,
                undersampling = self.undersampling,
                )
        self.session.add(self.run_config)
        self.session.commit()

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        y_true, y_pred = self._format_y(self.y), self._format_y(y_pred)
        mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        tn, fp, fn, tp = list(map(int, mat.ravel()))

        epoch_stats = EpochStats(
                run_id=self.run_id,
                epoch_id=epoch,
                true_positive = tp,
                true_negative = tn,
                false_positive = fp,
                false_negative = fn,
                )
        self.session.add(epoch_stats)
        self.session.commit()


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
