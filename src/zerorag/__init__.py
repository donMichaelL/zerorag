import logging

# Prevent "No handler found" warnings for users who don't configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
