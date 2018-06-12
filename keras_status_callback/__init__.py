from keras.callbacks import Callback
import sqlalchemy as sa



class StatusCallback(Callback):
    def __init__(self, db_connection_string):
        self.engine = sa.create_engine(db_connection_string)


