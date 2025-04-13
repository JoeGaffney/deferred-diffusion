import logging
import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create a logger object
logger = logging.getLogger(__name__)


def log_pretty(message, obj):
    """Utility function to pretty print objects in logs."""
    logger.info(message + "\n%s", pprint.pformat(obj, indent=1, width=120, sort_dicts=False))
