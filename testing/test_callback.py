import keras_status_callback as K
import sqlite3
import pytest
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv('.env')


@pytest.fixture
def user():
    return os.environ['TEST_USER']


@pytest.fixture
def password():
    return os.environ['TEST_PASSWORD']


@pytest.fixture
def dbname():
    return os.environ['TEST_DB']


@pytest.fixture
def dbhost():
    return os.environ['DB_HOST']


@pytest.fixture
def connection_string(user, dbname, dbhost, password):
    return f'postgres+psycopg2://{user}:{password}@{dbhost}/{dbname}'


@pytest.fixture
def run_id():
    return 42


@pytest.fixture
def callback(connection_string, run_id):
    return K.StatusCallback(run_id, connection_string, verbose=True, reset=True)


def test_creating_tables(callback, user, dbname):
    with psycopg2.connect(user=user, database=dbname) as con:
        cur = con.cursor()
        for tablename in ['run_configurations', 'epoch_stats']:
            cur.execute('select * from {}'.format(tablename))

    # No exception is good

def test_not_setting_data_raises_exception(callback):
    with pytest.raises(ValueError) as exc:
        callback.on_train_begin()

    assert 'set_data' in str(exc)


def test_from_categorical_two_classes():
    y = np.array([[0, 1], [1, 0]])
    assert (K.from_categorical(y) == np.array([1, 0])).all()

def test_from_categorical_four_classes():
    y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert (K.from_categorical(y) == np.array([0, 1, 2, 3])).all()
