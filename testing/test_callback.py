from keras_status_callback import StatusCallback


def test_creating_callback():
    callback = StatusCallback("sqlite:///tmp/db.db")
