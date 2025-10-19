import json

from src.config.paths import FAQS_PATH, FAQS_IDX_PATH, GROUND_TRUTH_PATH, GROUND_TRUTH_IDX_PATH

def load_FAQS(idx=False):
	path = FAQS_IDX_PATH if idx else FAQS_PATH
	with open(path) as f:
		faqs = json.load(f)

	return faqs

def load_ground_truth(idx=False):
	path = GROUND_TRUTH_IDX_PATH if idx else GROUND_TRUTH_PATH
	with open(path) as f:
		ground_truth = json.load(f)

	return ground_truth

