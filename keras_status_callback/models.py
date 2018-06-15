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
import datetime

Base = declarative_base()

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