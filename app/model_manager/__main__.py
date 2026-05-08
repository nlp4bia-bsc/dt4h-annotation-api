import logging
from . import ModelManager
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


if __name__ == "__main__":
    manager = ModelManager()
    manager.sanitize()